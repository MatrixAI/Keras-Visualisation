{
  pkgs ? import ./pkgs.nix,
  pythonPath ? "python35"
}:
  with pkgs;
  let
    python = lib.getAttrFromPath (lib.splitString "." pythonPath) pkgs;
  in
    python.pkgs.buildPythonApplication {
      name = "keras-visualisation";
      pname = "keras-visualisation";
      version = "0.0.1";
      src = lib.cleanSource ./.;
      propagatedBuildInputs =  (with python.pkgs; [
        pillow
        numpy
        tensorflowWithCuda
        Keras
        h5py
        (matplotlib.override { enableQt = true; })
      ]) ++ [ cudatoolkit ];
      KERAS_BACKEND = "tensorflow";
      MPLBACKEND = "Qt4Agg";
    }
