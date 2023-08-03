with import <nixpkgs> {};

mkShell {
  nativeBuildInputs = [
    pkg-config 
    cmake 
    gcc
    gnumake
  ];
  buildInputs = [
    # XORG ##
    xorg.libX11
    xorg.libXrandr
    xorg.libXrender
    xorg.libXScrnSaver
    xorg.libXext
    xorg.libXft
    xorg.libXpm.out
    xorg.libXrandr
    xorg.libXrender
    xorg.libXcursor
    xorg.xinput
    xorg.libXi
    xorg.libICE
    xorg.xorgproto
    xorg.libXinerama
    xorg.libXxf86vm
    ## END XORG ##
    ## WAYLAND ##
    ## END WAYLAND ##
    # vulkan for linux
    vulkan-headers
    vulkan-loader
    vulkan-tools
    vulkan-validation-layers
  ];

  VULKAN_SDK = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
}