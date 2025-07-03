

{ pkgs, ... }: {
  channel = "stable-24.05";

  packages = [
    pkgs.stdenv.cc.cc.lib
    pkgs.python311Packages.streamlit
    pkgs.python311Packages.pandas
    pkgs.python311Packages.matplotlib
    pkgs.python311Packages.seaborn
    pkgs.python311Packages.scikitlearn
    pkgs.python311Packages.joblib
  ];

  env = {
    PYTHONPATH = "$PWD";
  };
}