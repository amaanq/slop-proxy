{
  description = "slop-proxy: serve Anthropic/OpenAI API endpoints from Codex subscription accounts";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      forEachSystem =
        fn:
        nixpkgs.lib.genAttrs nixpkgs.lib.platforms.linux (
          system: fn system nixpkgs.legacyPackages.${system}
        );
    in
    {
      devShells = forEachSystem (
        system: pkgs: {
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              cargo
              rustc
              rustfmt
              clippy
              rust-analyzer
              sqlite
            ];
          };
        }
      );

      packages = forEachSystem (
        system: pkgs: {
          default = pkgs.rustPlatform.buildRustPackage {
            pname = "slop-proxy";
            version = "0.1.0";
            src = self;
            cargoLock.lockFile = ./Cargo.lock;
            meta = {
              description = "Anthropic/OpenAI API proxy backed by Codex subscription accounts";
              mainProgram = "slop-proxy";
            };
          };
        }
      );

      nixosModules.default =
        {
          config,
          lib,
          pkgs,
          ...
        }:
        let
          cfg = config.services.slop-proxy;
          settingsFormat = pkgs.formats.toml { };
          configFile = settingsFormat.generate "slop-proxy-config.toml" cfg.settings;
        in
        {
          options.services.slop-proxy = {
            enable = lib.mkEnableOption "slop-proxy";

            package = lib.mkOption {
              type = lib.types.package;
              default = self.packages.${pkgs.system}.default;
              defaultText = lib.literalExpression "slop-proxy.packages.\${pkgs.system}.default";
              description = "slop-proxy package to run.";
            };

            bind = lib.mkOption {
              type = lib.types.str;
              default = "[::1]:8484";
              description = "Address the API server listens on.";
            };

            settings = lib.mkOption {
              type = settingsFormat.type;
              default = { };
              description = "Contents of config.toml (codex.*, models.*).";
            };

            openFirewall = lib.mkOption {
              type = lib.types.bool;
              default = false;
              description = "Open the firewall for the bind port.";
            };
          };

          config = lib.mkIf cfg.enable {
            users.users.slop-proxy = {
              isSystemUser = true;
              group = "slop-proxy";
              home = "/var/lib/slop-proxy";
            };
            users.groups.slop-proxy = { };

            systemd.services.slop-proxy = {
              description = "slop-proxy API server";
              wantedBy = [ "multi-user.target" ];
              after = [ "network-online.target" ];
              wants = [ "network-online.target" ];
              serviceConfig = {
                User = "slop-proxy";
                Group = "slop-proxy";
                StateDirectory = "slop-proxy";
                ExecStart = "${lib.getExe cfg.package} --db /var/lib/slop-proxy/slop.db --config ${configFile} serve --bind ${cfg.bind}";
                Restart = "on-failure";
                RestartSec = 5;
                NoNewPrivileges = true;
                ProtectSystem = "strict";
                ProtectHome = true;
                PrivateTmp = true;
                PrivateDevices = true;
                ProtectKernelTunables = true;
                ProtectKernelModules = true;
                ProtectControlGroups = true;
                RestrictSUIDSGID = true;
                LockPersonality = true;
              };
            };

            networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [
              (lib.toInt (lib.last (lib.splitString ":" cfg.bind)))
            ];
          };
        };
    };
}
