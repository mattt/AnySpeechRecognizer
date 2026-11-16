import Foundation
import Testing

import AnySpeechRecognizer

private struct UpstreamError: LocalizedError, Equatable {
    var errorDescription: String? { "The upstream model stopped." }
}

private let errorCases: [(SpeechRecognitionError, SpeechRecognitionError.Kind, String)] = [
    (.unsupportedConfiguration("No such option."), .unsupportedConfiguration, "No such option."),
    (.unavailableProvider("No Neural Engine."), .unavailableProvider, "No Neural Engine."),
    (.unsupportedLanguage("No Klingon."), .unsupportedLanguage, "No Klingon."),
    (.missingAssets("Install the model."), .missingAssets, "Install the model."),
    (.assetVerificationFailed("model.bin"), .assetVerificationFailed, "model.bin"),
    (.invalidAudio("Empty audio."), .invalidAudio, "Empty audio."),
    (.invalidSessionState("Already finished."), .invalidSessionState, "Already finished."),
    (
        .bufferOverflow, .bufferOverflow,
        "The result buffer is full. Consume results during analysis, or cancel the analyzer."
    ),
    (.invalidResult("Bad timing."), .invalidResult, "Bad timing."),
    (
        .backendFailure(underlying: UpstreamError()), .backendFailure,
        "Speech recognition failed: The upstream model stopped."
    ),
]

private let kindDescriptions: [(SpeechRecognitionError.Kind, String)] = [
    (.unsupportedConfiguration, "Unsupported configuration"),
    (.unavailableProvider, "Unavailable provider"),
    (.unsupportedLanguage, "Unsupported language"),
    (.missingAssets, "Missing assets"),
    (.assetVerificationFailed, "Asset verification failed"),
    (.invalidAudio, "Invalid audio"),
    (.invalidSessionState, "Invalid session state"),
    (.bufferOverflow, "Buffer overflow"),
    (.invalidResult, "Invalid result"),
    (.backendFailure, "Backend failure"),
]

@Test func errorAccessorsDescribeEveryCase() {
    for (error, kind, message) in errorCases {
        #expect(error.kind == kind)
        #expect(error.message == message)
        #expect(error.errorDescription == message)
        #expect(error.errorDescription == error.message)
        #expect(error.localizedDescription == message)
        if error.kind == .backendFailure {
            #expect(error.underlyingError as? UpstreamError == UpstreamError())
        } else {
            #expect(error.underlyingError == nil)
        }
    }
}

@Test func errorKindsAreDistinctAndDescribed() {
    #expect(kindDescriptions.count == errorCases.count)
    for (index, (kind, description)) in kindDescriptions.enumerated() {
        #expect(kind.description == description)
        #expect(kind == kind)
        for (otherIndex, (other, _)) in kindDescriptions.enumerated() where otherIndex != index {
            #expect(kind != other)
        }
    }
    #expect(Set(kindDescriptions.map(\.0)).count == kindDescriptions.count)
}

@Test func errorCaseConstructorsAreFunctionValues() {
    let typed: (String) -> SpeechRecognitionError = SpeechRecognitionError.missingAssets
    let untyped = SpeechRecognitionError.missingAssets
    #expect(typed("a").kind == .missingAssets)
    #expect(untyped("b").message == "b")

    let typedFailure: (any Error) -> SpeechRecognitionError =
        SpeechRecognitionError.backendFailure(underlying:)
    let untypedFailure = SpeechRecognitionError.backendFailure(underlying:)
    #expect(typedFailure(UpstreamError()).underlyingError is UpstreamError)
    #expect(untypedFailure(UpstreamError()).kind == .backendFailure)

    let constructors: [(String) -> SpeechRecognitionError] = [
        SpeechRecognitionError.unsupportedConfiguration,
        SpeechRecognitionError.unavailableProvider,
        SpeechRecognitionError.unsupportedLanguage,
        SpeechRecognitionError.missingAssets,
        SpeechRecognitionError.assetVerificationFailed,
        SpeechRecognitionError.invalidAudio,
        SpeechRecognitionError.invalidSessionState,
        SpeechRecognitionError.invalidResult,
    ]
    #expect(constructors.map { $0("reason").message } == Array(repeating: "reason", count: 8))
}

@Test func handlersMatchByKind() {
    var matched = false
    do {
        throw SpeechRecognitionError.missingAssets("Install the model.")
    } catch let error as SpeechRecognitionError where error.kind == .missingAssets {
        matched = error.message == "Install the model."
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
    #expect(matched)
}

@Test func handlersReadTheUnderlyingError() {
    var underlying: (any Error)?
    do {
        throw SpeechRecognitionError.backendFailure(underlying: UpstreamError())
    } catch let error as SpeechRecognitionError {
        underlying = error.underlyingError
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
    #expect(underlying as? UpstreamError == UpstreamError())
}

@Test func caseHandlersFromVersion01StillMatch() {
    var matchedOverflow = false
    do {
        throw SpeechRecognitionError.bufferOverflow
    } catch SpeechRecognitionError.bufferOverflow {
        matchedOverflow = true
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
    #expect(matchedOverflow)

    var underlying: (any Error)?
    do {
        throw SpeechRecognitionError.backendFailure(underlying: UpstreamError())
    } catch SpeechRecognitionError.backendFailure(let error) {
        underlying = error
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
    #expect(underlying is UpstreamError)
}
