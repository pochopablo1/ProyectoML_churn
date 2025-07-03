# Archivo de configuración para el entorno de desarrollo Nix
# Versión simplificada para máxima compatibilidad
{ pkgs, ... }: {
  # Define el canal de paquetes a usar
  channel = "stable-24.05";

  # 1. Lista de paquetes a instalar en el entorno
  packages = [
    # Librería del sistema necesaria para algunas dependencias de Python
    pkgs.stdenv.cc.cc.lib

    # Paquetes de Python requeridos por el proyecto
    pkgs.python311Packages.streamlit
    pkgs.python311Packages.pandas
    pkgs.python311Packages.matplotlib
    pkgs.python311Packages.seaborn
    pkgs.python311Packages.scikitlearn
    pkgs.python311Packages.joblib
  ];

  # 2. Configuración de variables de entorno
  env = {
    # Esta línea soluciona el error de importación de 'utils.functions'
    # Le dice a Python que busque módulos desde la carpeta raíz del proyecto.
    PYTHONPATH = "$PWD"; 
  };
}
