import AVFoundation
import Speech

/// Reserves a locale, checks module assets, installs them, and releases the reservation.
func prepareAssets(for locale: Locale) async throws -> Bool {
    guard
        let supported = await DictationTranscriber.supportedLocale(equivalentTo: locale),
        await SpeechTranscriber.supportedLocales.contains(supported)
    else { return false }
    let reserved = await AssetInventory.reservedLocales
    if reserved.count >= AssetInventory.maximumReservedLocales, let oldest = reserved.first {
        await AssetInventory.release(reservedLocale: oldest)
    }
    try await AssetInventory.reserve(locale: supported)
    let modules: [any SpeechModule] = [
        SpeechTranscriber(locale: supported, preset: .timeIndexedTranscriptionWithAlternatives),
        DictationTranscriber(
            locale: supported, contentHints: [.shortForm], transcriptionOptions: [.punctuation],
            reportingOptions: [.alternativeTranscriptions],
            attributeOptions: [.audioTimeRange, .transcriptionConfidence]),
        SpeechDetector(),
    ]
    if await AssetInventory.status(forModules: modules) < .installed,
        let request = try await AssetInventory.assetInstallationRequest(supporting: modules)
    {
        print(request.progress.fractionCompleted)
        try await request.downloadAndInstall()
    }
    print(await SpeechTranscriber.installedLocales.count, SpeechTranscriber.isAvailable)
    return await AssetInventory.status(forModules: modules) == .installed
}

@main enum AssetReservationExample {
    static func main() async throws {
        guard CommandLine.arguments.count == 2 else {
            print("Usage: AssetReservationExample <locale>")
            return
        }
        print(try await prepareAssets(for: Locale(identifier: CommandLine.arguments[1])))
    }
}
