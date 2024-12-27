// swift-tools-version: 5.8
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "stable-diffusion",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(
            name: "StableDiffusion",
            targets: ["StableDiffusion"]),
        .executable(
            name: "StableDiffusionSample",
            targets: ["StableDiffusionCLI"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.3"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "0.1.8"),
        .package(url: "https://github.com/pvieito/PythonKit.git", branch: "master")
    ],
    targets: [
        .target(
            name: "StableDiffusion",
            dependencies:  [
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "PythonKit", package: "PythonKit"),
            ],
            path: "swift/StableDiffusion"),
        .executableTarget(
            name: "StableDiffusionCLI",
            dependencies: [
                "StableDiffusion",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "PythonKit", package: "PythonKit")],
            path: "swift/StableDiffusionCLI"),
        .testTarget(
            name: "StableDiffusionTests",
            dependencies: ["StableDiffusion"],
            path: "swift/StableDiffusionTests",
            resources: [
                .copy("Resources/vocab.json"),
                .copy("Resources/merges.txt")
            ]),
    ]
)
