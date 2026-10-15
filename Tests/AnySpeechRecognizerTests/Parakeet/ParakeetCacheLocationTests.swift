import Foundation
import Testing

@testable import AnySpeechRecognizer

struct ParakeetCacheLocationTests {
    let support = URL(fileURLWithPath: "/tmp/support", isDirectory: true)
    let home = URL(fileURLWithPath: "/tmp/home", isDirectory: true)

    func resolve(_ environment: [String: String]) -> String {
        ModelAssetStore.defaultCacheDirectory(
            environment: environment, applicationSupport: support, home: home
        ).standardizedFileURL.path
    }

    @Test func environmentVariablesTakePrecedence() {
        #expect(resolve(["HF_HUB_CACHE": "/tmp/hub", "HF_HOME": "/tmp/hf"]) == "/tmp/hub")
        #expect(resolve(["HF_HOME": "/tmp/hf"]) == "/tmp/hf/hub")
    }

    @Test func sandboxedAppsUseApplicationSupport() {
        #expect(
            resolve(["APP_SANDBOX_CONTAINER_ID": "app"])
                == "/tmp/support/AnySpeechRecognizer/HuggingFace/hub")
    }

    #if os(macOS)
        @Test func unsandboxedMacsShareTheStandardCache() {
            #expect(resolve([:]) == "/tmp/home/.cache/huggingface/hub")
        }
    #endif
}
