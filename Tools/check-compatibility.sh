#!/usr/bin/env bash
# Builds each compatibility fixture against Speech, then against AnySpeechRecognizer
# with only its import changed. Launches every executable without arguments to check
# linking without installing assets.
set -euo pipefail

cd "$(dirname "$0")/.."
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/anyspeech-compatibility.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

target_arch="$(uname -m)"
# SwiftPM names a path dependency after its directory, which differs in worktrees.
package_identity="$(basename "$PWD" | tr '[:upper:]' '[:lower:]')"
package_dir="$temporary_dir/Package"
targets=""

for fixture in Tests/Fixtures/Compatibility/*.swift; do
    name="$(basename "$fixture" .swift)"
    echo "Checking $name against Speech"
    xcrun swiftc -swift-version 6 -target "$target_arch-apple-macos26.0" \
        -parse-as-library "$fixture" -o "$temporary_dir/Native$name"

    mkdir -p "$package_dir/Sources/$name"
    if ! grep -q '^import Speech$' "$fixture"; then
        echo "error: $fixture must import Speech on its own line" >&2
        exit 1
    fi
    sed 's/^import Speech$/import AnySpeechRecognizer/' "$fixture" \
        > "$package_dir/Sources/$name/$name.swift"
    targets="$targets
        .executableTarget(
            name: \"$name\",
            dependencies: [.product(name: \"AnySpeechRecognizer\", package: \"$package_identity\")]),"
done

cat > "$package_dir/Package.swift" <<MANIFEST
// swift-tools-version: 6.2
import PackageDescription
let package = Package(
    name: "CompatibilityFixtures",
    platforms: [.macOS(.v26)],
    dependencies: [.package(path: "$PWD")],
    targets: [$targets
    ],
    swiftLanguageModes: [.v6]
)
MANIFEST

echo "Checking fixtures against AnySpeechRecognizer"
xcrun swift build --package-path "$package_dir"
fixture_dir="$(xcrun swift build --package-path "$package_dir" --show-bin-path)"

for fixture in Tests/Fixtures/Compatibility/*.swift; do
    name="$(basename "$fixture" .swift)"
    "$temporary_dir/Native$name"
    "$fixture_dir/$name"
done
