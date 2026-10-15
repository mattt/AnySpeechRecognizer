import Foundation
import Speech

/// Asset status, explicit installation, and locale reservations for package modules.
///
/// Apple-backed modules use Apple's native inventory.
/// Model-backed modules use their model's asset store.
public enum AssetInventory {
    /// The combined asset state of the selected modules.
    public enum Status: Int, Sendable, Hashable, Comparable {
        /// At least one module cannot run on this device or locale.
        case unsupported
        /// Assets must be installed before use.
        case supported
        /// Installation or preparation is in progress.
        case downloading
        /// All required assets are local.
        case installed
        public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }
    }

    /// The maximum number of locales Apple allows an app to reserve.
    public static var maximumReservedLocales: Int { Speech.AssetInventory.maximumReservedLocales }

    /// The locales this app has reserved with Apple's inventory.
    public static var reservedLocales: [Locale] {
        get async { await Speech.AssetInventory.reservedLocales }
    }

    /// Reserves Apple speech assets for a locale.
    /// Third-party models have no locale reservations.
    /// - Parameter locale: The locale to reserve.
    /// - Returns: Whether a new reservation was made.
    /// - Throws: An error if the reservation limit is reached or the locale is unsupported.
    @discardableResult
    public static func reserve(locale: Locale) async throws -> Bool {
        try await Speech.AssetInventory.reserve(locale: locale)
    }

    /// Releases a reservation made with ``reserve(locale:)``.
    /// - Parameter reservedLocale: The reserved locale.
    /// - Returns: Whether a reservation was released.
    @discardableResult
    public static func release(reservedLocale: Locale) async -> Bool {
        await Speech.AssetInventory.release(reservedLocale: reservedLocale)
    }

    /// Checks asset state without starting a download.
    ///
    /// Model-backed modules report the assets that their own request uses,
    /// such as the Parakeet live encoder for volatile or fast results.
    /// - Parameter modules: Package modules.
    /// - Returns: The least-ready state among the modules.
    public static func status(forModules modules: [any SpeechModule]) async -> Status {
        guard let groups = try? ModuleGroups(modules) else { return .unsupported }
        var status = Status.installed
        if !groups.native.isEmpty {
            status = min(
                status, Status(await Speech.AssetInventory.status(forModules: groups.native)))
        }
        for (model, request) in groups.models {
            guard (try? request.validate(for: model.capabilities)) != nil else {
                return .unsupported
            }
            status = min(status, Status(await model.availability(for: request)))
        }
        return status
    }

    /// Creates one installation request for every module that needs assets.
    ///
    /// Each model receives the requests of all of its modules in one call,
    /// so assets that several modules share install once.
    /// - Parameter modules: Package modules.
    /// - Returns: A request, or `nil` when assets are already installed.
    /// - Throws: A configuration, availability, or backend error.
    public static func assetInstallationRequest(supporting modules: [any SpeechModule])
        async throws -> AssetInstallationRequest?
    {
        let groups = try ModuleGroups(modules)
        var requests: [AssetInstallationRequest] = []
        if !groups.native.isEmpty,
            let native = try await Speech.AssetInventory.assetInstallationRequest(
                supporting: groups.native)
        {
            requests.append(AssetInstallationRequest(native: native))
        }
        for (model, modelRequests) in groups.requestsByModel {
            for request in modelRequests { try request.validate(for: model.capabilities) }
            if let installation = try await model.assetInstallationRequest(
                supporting: modelRequests)
            {
                requests.append(installation)
            }
        }
        if requests.count <= 1 { return requests.first }
        return AssetInstallationRequest(combining: requests)
    }
}

extension AssetInventory.Status {
    init(_ native: Speech.AssetInventory.Status) {
        switch native {
        case .installed: self = .installed
        case .downloading: self = .downloading
        case .supported: self = .supported
        default: self = .unsupported
        }
    }

    init(_ availability: SpeechRecognitionAvailability) {
        switch availability {
        case .ready: self = .installed
        case .missingAssets: self = .supported
        case .preparing: self = .downloading
        case .unsupportedHardware, .unsupportedLanguage: self = .unsupported
        }
    }
}

/// Package modules split into native modules and model-backed requests.
struct ModuleGroups {
    var native: [any Speech.SpeechModule] = []
    var models: [(any SpeechRecognitionModel, SpeechRecognitionRequest)] = []

    init(_ modules: [any SpeechModule]) throws {
        for module in modules {
            guard let module = module as? any AnalyzerModule else {
                throw SpeechRecognitionError.unsupportedConfiguration(
                    "\(type(of: module)) is not a package speech module.")
            }
            switch module.binding.backend {
            case .native(let member): native.append(member.module)
            case .model(let model, let request): models.append((model, request))
            }
        }
    }

    /// The distinct requests of each model, in module order, keyed by model identifier.
    /// Modules whose models share an identifier use the first module's model.
    var requestsByModel: [(any SpeechRecognitionModel, [SpeechRecognitionRequest])] {
        var groups: [(any SpeechRecognitionModel, [SpeechRecognitionRequest])] = []
        var indices: [String: Int] = [:]
        for (model, request) in models {
            if let index = indices[model.modelID] {
                if !groups[index].1.contains(request) { groups[index].1.append(request) }
            } else {
                indices[model.modelID] = groups.count
                groups.append((model, [request]))
            }
        }
        return groups
    }
}

/// An explicit network boundary for installing model assets.
/// Providers can share installation work while retaining independent waiter cancellation.
public final class AssetInstallationRequest: NSObject, ProgressReporting, Sendable {
    /// Download progress, independent of transcription progress.
    public let progress: Progress
    private let install: @Sendable () async throws -> Void

    /// Creates an installation request for a provider.
    /// - Parameters:
    ///   - progress: The provider's thread-safe progress object.
    ///   - downloadAndInstall: The operation to run only on explicit installation.
    public init(progress: Progress, downloadAndInstall: @escaping @Sendable () async throws -> Void)
    {
        self.progress = progress
        self.install = downloadAndInstall
    }

    convenience init(native: Speech.AssetInstallationRequest) {
        self.init(progress: native.progress) { try await native.downloadAndInstall() }
    }

    convenience init(combining requests: [AssetInstallationRequest]) {
        let progress = Progress(totalUnitCount: Int64(requests.count) * 100)
        for request in requests {
            progress.addChild(request.progress, withPendingUnitCount: 100)
        }
        self.init(progress: progress) {
            try await withThrowingTaskGroup(of: Void.self) { group in
                for request in requests { group.addTask { try await request.downloadAndInstall() } }
                try await group.waitForAll()
            }
        }
    }

    /// Installs the requested assets.
    /// - Throws: A backend error or `CancellationError`.
    public func downloadAndInstall() async throws {
        try Task.checkCancellation()
        do {
            try await install()
            try Task.checkCancellation()
        } catch { throw recognitionError(error) }
    }
}
