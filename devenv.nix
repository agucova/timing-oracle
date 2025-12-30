{ pkgs, lib, ... }:

{
  languages.rust = {
    enable = true;
    channel = "stable";
    components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" "rust-src" ];
  };

  packages = with pkgs; [
    cargo-nextest
    cargo-edit
  ];

  env = {
    RUST_BACKTRACE = "1";
  };

  enterShell = ''
    command -v cargo-llvm-cov &> /dev/null || cargo install cargo-llvm-cov --quiet
    command -v cargo-nextest &> /dev/null || cargo install cargo-nextest --quiet
  '';
}
