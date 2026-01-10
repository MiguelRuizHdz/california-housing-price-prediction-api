#!/bin/bash

set -e

echo "=========================================="
echo "California Housing ML Project - Setup"
echo "=========================================="
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Funcion para imprimir mensajes
print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_info() {
    echo "[INFO] $1"
}

# Detectar Python compatible (3.9-3.12)
detect_python() {
    local python_cmd=""
    
    for cmd in python3.12 python3.11 python3.10 python3.9 python3 python; do
        if command -v $cmd &> /dev/null; then
            local version=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            local major=$(echo $version | cut -d. -f1)
            local minor=$(echo $version | cut -d. -f2)
            
            if [ "$major" -eq 3 ] && [ "$minor" -ge 9 ] && [ "$minor" -le 12 ]; then
                python_cmd=$cmd
                echo $python_cmd
                return 0
            fi
        fi
    done
    
    return 1
}

# Verificar si UV esta instalado
check_uv() {
    if command -v uv &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Main setup
main() {
    print_info "Detectando Python compatible..."
    
    PYTHON_CMD=$(detect_python)
    
    if [ -z "$PYTHON_CMD" ]; then
        print_error "No se encontro Python 3.9-3.12"
        print_info "Por favor instala Python 3.11 o 3.12:"
        echo "  - macOS: brew install python@3.11"
        echo "  - Linux: sudo apt install python3.11"
        echo "  - O usa pyenv: pyenv install 3.11.7"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    print_success "Python $PYTHON_VERSION detectado en: $(which $PYTHON_CMD)"
    
    # Preguntar por gestor de paquetes
    echo ""
    if check_uv; then
        print_success "UV detectado: $(which uv)"
        echo ""
        read -p "Usar UV para instalar dependencias? [S/n]: " use_uv
        use_uv=${use_uv:-S}
    else
        print_warning "UV no esta instalado"
        echo ""
        read -p "Instalar UV (mas rapido)? [S/n]: " install_uv
        install_uv=${install_uv:-S}
        
        if [[ $install_uv =~ ^[Ss]$ ]]; then
            print_info "Instalando UV..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            
            # Agregar UV al PATH para esta sesion
            export PATH="$HOME/.cargo/bin:$PATH"
            
            if check_uv; then
                print_success "UV instalado correctamente"
                use_uv="S"
            else
                print_warning "No se pudo instalar UV, usando pip"
                use_uv="n"
            fi
        else
            use_uv="n"
        fi
    fi
    
    echo ""
    print_info "Creando ambiente virtual..."
    
    if [ -d "venv" ]; then
        read -p "El ambiente virtual ya existe. Recrear? [s/N]: " recreate
        recreate=${recreate:-N}
        
        if [[ $recreate =~ ^[Ss]$ ]]; then
            print_info "Eliminando ambiente virtual existente..."
            rm -rf venv
        else
            print_info "Usando ambiente virtual existente"
        fi
    fi
    
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_success "Ambiente virtual creado"
    fi
    
    # Activar ambiente virtual
    print_info "Activando ambiente virtual..."
    source venv/bin/activate
    
    # Actualizar pip
    print_info "Actualizando pip..."
    python -m pip install --upgrade pip --quiet
    
    # Instalar dependencias
    echo ""
    if [[ $use_uv =~ ^[Ss]$ ]]; then
        print_info "Instalando dependencias con UV (mas rapido)..."
        uv pip install -r requirements.txt
    else
        print_info "Instalando dependencias con pip..."
        pip install -r requirements.txt --quiet
    fi
    
    print_success "Dependencias instaladas"
    
    # Crear directorios necesarios
    echo ""
    print_info "Creando estructura de directorios..."
    mkdir -p data models outputs mlruns logs
    print_success "Directorios creados"
    
    # Copiar .env.example si no existe .env
    if [ ! -f ".env" ]; then
        print_info "Creando archivo .env..."
        cp .env.example .env
        print_success "Archivo .env creado"
    else
        print_info "Archivo .env ya existe"
    fi
    
    # Verificar instalacion
    echo ""
    print_info "Verificando instalacion..."
    
    if python -c "import sklearn, fastapi, mlflow, pandas, numpy" 2>/dev/null; then
        print_success "Todas las dependencias estan instaladas correctamente"
    else
        print_error "Algunas dependencias no se instalaron correctamente"
        exit 1
    fi
    
    # Resumen
    echo ""
    echo "=========================================="
    echo "Setup completado exitosamente"
    echo "=========================================="
    echo ""
    print_success "Ambiente: venv/"
    print_success "Python: $PYTHON_VERSION"
    print_success "Gestor de paquetes: $([ $use_uv == 'S' ] && echo 'UV' || echo 'pip')"
    echo ""
    echo "Proximos pasos:"
    echo ""
    echo "1. Activar ambiente virtual:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Entrenar modelo (elige una opcion):"
    echo "   a) Notebook: jupyter lab notebooks/01_california_housing_pipeline.ipynb"
    echo "   b) Script: Ejecuta el comando de entrenamiento del README"
    echo ""
    echo "3. Iniciar API:"
    echo "   uvicorn app.main:app --reload"
    echo ""
    echo "4. Acceder a documentacion:"
    echo "   http://localhost:8000/docs"
    echo ""
    echo "Para mas informacion, consulta README.md o QUICKSTART.md"
    echo ""
}

# Ejecutar setup
main

