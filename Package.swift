// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "AnySpeechRecognizer",
    platforms: [.macOS(.v26), .iOS(.v26)],
    products: [
        .library(name: "AnySpeechRecognizer", targets: ["AnySpeechRecognizer"]),
        .executable(name: "hear", targets: ["hear"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.8.2"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "AnySpeechRecognizer",
            dependencies: [.product(name: "HuggingFace", package: "swift-huggingface")]
        ),
        .executableTarget(
            name: "hear",
            dependencies: [
                "AnySpeechRecognizer",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/CLI"
        ),
        .testTarget(
            name: "AnySpeechRecognizerTests",
            dependencies: [
                "AnySpeechRecognizer", .product(name: "HuggingFace", package: "swift-huggingface"),
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
