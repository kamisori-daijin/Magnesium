// swift-tools-version: 6.4
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MagnesiumKit",
    platforms: [
        .macOS(.v27)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "MagnesiumKit",
            targets: ["MagnesiumKit"],

        ),
    
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "MagnesiumKit",
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
            resources: [
                .process("Resources/ane_texture_processor.aimodel"),
                .process("Resources/ane_3d_rasterizer.aimodel"),
                .process("Resources/ane_pre_processor.aimodel")
            ],
        ),
        .testTarget(
            name: "MagnesiumKitTests",
            dependencies: ["MagnesiumKit"],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
    ]
)
