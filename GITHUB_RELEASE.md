# Task487 UMI deployment source snapshot

This private repository is a source-only snapshot of the Task487 Marvin
deployment workspace prepared on 2026-08-19.

Included:

- Task487 policy client, runtime contract, scheduler, diagnostics, and tests;
- the locally modified OpenPI source required by the server;
- geometry-mask projection and manual calibration source/configuration;
- UMI robot client dependencies, camera utilities, and offline audit scripts;
- current runbooks and investigation reports.

Intentionally excluded:

- policy checkpoints and model weights;
- robot datasets and captured videos;
- runtime logs and generated evaluation arrays;
- Python environments, caches, compiled helper binaries, and credentials.

The commands in `TASK487_CURRENT_RUN_COMMANDS.md` retain the original absolute
paths used on the development machine. Restore checkpoints separately before
starting a policy server.

## 2026-09-05 client update

Archives the four-wrist deployment contracts, author-sync scheduler and speed
limits, passive gripper/policy I/O logging, paired physical/policy gripper
compensation, first right-to-left handoff gate, and open-before-close HOME
preparation. Associated source tests and investigation reports are included.
Reports retain development-machine evidence paths; raw evidence is not bundled.
The external Marvin/Mink ROS backend is a separate installation.
