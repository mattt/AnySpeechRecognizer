// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "AnySpeechRecognizer",
    platforms: [.macOS(.v26), .iOS(.v26)],
    products: [
        .library(name: "AnySpeechRecognizer", targets: ["AnySpeechRecognizer"]),
    ],
    targets: [
        .target(name: "AnySpeechRecognizer"),
        .testTarget(
            name: "AnySpeechRecognizerTests",
            dependencies: ["AnySpeechRecognizer"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
