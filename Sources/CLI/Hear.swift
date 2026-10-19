import AnySpeechRecognizer
import ArgumentParser
import Darwin
import Foundation

typealias SpeechTranscriber = AnySpeechRecognizer.SpeechTranscriber
typealias SpeechAnalyzer = AnySpeechRecognizer.SpeechAnalyzer
typealias AssetInventory = AnySpeechRecognizer.AssetInventory
typealias SpeechModule = AnySpeechRecognizer.SpeechModule

@main
struct Hear: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "hear",
        abstract:
            "Recognize speech in local audio files with Apple, Parakeet, or Qwen3-ASR models.",
        discussion: "SIGINT and SIGTERM cancel the operation and wait for analyzer cleanup.",
        subcommands: [Status.self, Prepare.self, Transcribe.self, Evaluate.self]
    )
}

/// A subcommand whose operation SIGINT and SIGTERM cancel.
///
/// Cancellation prints "Cancelled." and exits with status 130;
/// other errors print `error: <description>` and exit with status 1.
protocol CancellableCommand: AsyncParsableCommand, Sendable {
    func execute() async throws
}

extension CancellableCommand {
    mutating func run() async throws {
        let command = self
        let operation = Task { try await command.execute() }
        signal(SIGINT, SIG_IGN)
        signal(SIGTERM, SIG_IGN)
        let signals = [SIGINT, SIGTERM].map { number in
            let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
            source.setEventHandler { @Sendable in operation.cancel() }
            source.activate()
            return source
        }
        defer { for source in signals { source.cancel() } }
        do { try await operation.value } catch is CancellationError {
            FileHandle.standardError.write(Data("Cancelled.\n".utf8))
            throw ExitCode(130)
        } catch {
            FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
            throw ExitCode.failure
        }
    }
}

struct CLIError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

func writeJSON(_ value: some Encodable) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    var data = try encoder.encode(value)
    data.append(10)
    FileHandle.standardOutput.write(data)
}
