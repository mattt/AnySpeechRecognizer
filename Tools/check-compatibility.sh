#!/usr/bin/env bash
# Builds each compatibility fixture against Speech, then against AnySpeechRecognizer
# with only its import changed, with and without MemberImportVisibility.
# Launches every executable without arguments to check linking without installing assets.
# Then checks that a file that imports both modules fails to compile with an ambiguity error.
set -euo pipefail

cd "$(dirname "$0")/.."
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/anyspeech-compatibility.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

target_arch="$(uname -m)"
# SwiftPM names a path dependency after its directory, which differs in worktrees.
package_identity="$(basename "$PWD" | tr '[:upper:]' '[:lower:]')"
package_dir="$temporary_dir/Package"
member_import_suffix="MemberImportVisibility"
targets=""

add_target() {
    local name="$1"
    local settings="$2"
    targets="$targets
        .executableTarget(
            name: \"$name\",
            dependencies: [.product(name: \"AnySpeechRecognizer\", package: \"$package_identity\")]$settings),"
}

write_manifest() {
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
}

for fixture in Tests/Fixtures/Compatibility/*.swift; do
    name="$(basename "$fixture" .swift)"
    echo "Checking $name against Speech"
    xcrun swiftc -swift-version 6 -target "$target_arch-apple-macos26.0" \
        -enable-upcoming-feature MemberImportVisibility \
        -parse-as-library "$fixture" -o "$temporary_dir/Native$name"

    if ! grep -q '^import Speech$' "$fixture"; then
        echo "error: $fixture must import Speech on its own line" >&2
        exit 1
    fi
    for target in "$name" "$name$member_import_suffix"; do
        mkdir -p "$package_dir/Sources/$target"
        sed 's/^import Speech$/import AnySpeechRecognizer/' "$fixture" \
            > "$package_dir/Sources/$target/$name.swift"
    done
    add_target "$name" ""
    add_target "$name$member_import_suffix" ",
            swiftSettings: [.enableUpcomingFeature(\"MemberImportVisibility\")]"
done

write_manifest
echo "Checking fixtures against AnySpeechRecognizer"
xcrun swift build --package-path "$package_dir"
fixture_dir="$(xcrun swift build --package-path "$package_dir" --show-bin-path)"

for fixture in Tests/Fixtures/Compatibility/*.swift; do
    name="$(basename "$fixture" .swift)"
    "$temporary_dir/Native$name"
    "$fixture_dir/$name"
    "$fixture_dir/$name$member_import_suffix"
done

# Depending on the compiler, names that both modules declare are ambiguous in a file
# that imports both, or resolve to this package's types.
# The typed line compiles only if SpeechAnalyzer resolves to the package's type.
echo "Checking a file that imports Speech and AnySpeechRecognizer"
mkdir -p "$package_dir/Sources/BothImports"
cat > "$package_dir/Sources/BothImports/main.swift" <<'SOURCE'
import AnySpeechRecognizer
import Speech

print(SpeechAnalyzer.self)
let _: AnySpeechRecognizer.SpeechAnalyzer.Type = SpeechAnalyzer.self
SOURCE
add_target "BothImports" ""
write_manifest
both_imports_log="$temporary_dir/BothImports.log"
if xcrun swift build --package-path "$package_dir" --target BothImports > "$both_imports_log" 2>&1; then
    echo "Shared names resolve to AnySpeechRecognizer's types in a file that imports both modules"
elif grep -q "error: .*ambiguous" "$both_imports_log"; then
    echo "Shared names are ambiguous in a file that imports both modules"
else
    cat "$both_imports_log" >&2
    echo "error: a file that imports Speech and AnySpeechRecognizer failed" \
        "without an ambiguity error; if SpeechAnalyzer resolves to Speech's type," \
        "update the README's import advice" >&2
    exit 1
fi
