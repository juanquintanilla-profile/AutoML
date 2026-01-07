"""
Streamlit UI for AutoML Agent.
Professional interface for automated machine learning.
"""

import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

# API Configuration
API_URL = "http://localhost:8000"

# Translations
TRANSLATIONS = {
    "en": {
        # Navigation
        "nav_new_job": "New Job",
        "nav_monitor": "Job Monitor",
        "nav_results": "Results",
        "nav_about": "About",
        "navigation": "Navigation",
        "language": "Language",
        "quick_info": "Quick Info",
        "quick_info_text": """**AutoML Agent** uses LLM-powered
orchestration to automatically:
- Analyze your data
- Select best models
- Optimize hyperparameters
- Evaluate performance""",

        # Main
        "title": "AutoML Agent",
        "subtitle": "Automated Machine Learning with Multi-Agent Orchestration",

        # New Job
        "create_new_job": "Create New AutoML Job",
        "upload_dataset": "Upload Your Dataset",
        "drop_csv": "Drop your CSV file here",
        "or_use_sample": "Or Use a Sample Dataset",
        "test_system": "Test the system with pre-configured datasets:",
        "select_sample": "-- Select --",
        "target": "Target",
        "dataset_preview": "Dataset Preview",
        "rows": "Rows",
        "columns": "Columns",
        "numeric": "Numeric",
        "missing_values": "Missing Values",
        "target_column": "Target Column (variable to predict)",
        "task_type": "Task Type",
        "classification": "Classification",
        "regression": "Regression",
        "advanced_options": "Advanced Options",
        "config_path": "Config Path",
        "start_job": "Start AutoML Job",
        "creating_job": "Creating job...",
        "job_created": "Job created successfully!",
        "job_id": "Job ID",
        "go_to_monitor": "Go to **Job Monitor** to track progress",
        "api_error": "Cannot connect to API. Make sure the API server is running:",
        "dataset_not_found": "Dataset not found",
        "error_loading": "Error loading dataset",

        # Monitor
        "job_monitor": "Job Monitor",
        "no_jobs": "No jobs found. Create a new job to get started!",
        "total_jobs": "Total Jobs",
        "running": "Running",
        "completed": "Completed",
        "failed": "Failed",
        "all_jobs": "All Jobs",
        "status": "Status",
        "created": "Created",
        "error": "Error",
        "job_details": "Job Details",
        "select_job": "Select a job to view details",
        "duration": "Duration",
        "iteration": "Iteration",
        "view_logs": "View Logs",
        "no_logs": "No logs available",
        "auto_refresh": "Auto-refreshing every 5 seconds...",

        # Results
        "results": "Results",
        "no_completed": "No completed jobs yet. Check Job Monitor for running jobs.",
        "select_completed": "Select Completed Job",
        "best_model": "Best Model",
        "algorithm": "Algorithm",
        "training_time": "Training Time",
        "model_details": "Model Details",
        "key_strengths": "Key Strengths",
        "auto_selected": "Automatically selected as the best performing model for your dataset.",
        "optimized_hyperparams": "Optimized Hyperparameters",
        "parameter": "Parameter",
        "value": "Value",
        "hyperparams_flaml": "Hyperparameters were optimized using FLAML's AutoML search.",
        "hyperparams_auto": "Hyperparameters were automatically tuned. Check run_summary.json for details.",
        "all_metrics": "All Evaluation Metrics",
        "download_model": "Download Model",
        "download_button": "Download Trained Model (.joblib)",
        "usage": "Usage",

        # About
        "about_title": "About AutoML Agent",
        "what_is_automl": "What is AutoML?",
        "automl_desc": """Automated Machine Learning (AutoML) is the process of automating the end-to-end process of applying
machine learning to real-world problems. Instead of manually selecting algorithms, tuning hyperparameters,
and engineering features, AutoML systems automatically search for the best configuration for your data.""",
        "multi_agent": "Multi-Agent Architecture",
        "multi_agent_desc": """This system uses a **multi-agent approach** where specialized AI agents collaborate to solve the machine learning problem.
An LLM (Large Language Model) acts as the orchestrator, deciding which agent to invoke next based on the current state.""",
        "workflow": "Workflow",
        "workflow_desc": """The Planner Agent orchestrates the entire workflow, ensuring each step completes before moving to the next.
It can decide to iterate (try more models or hyperparameters) or finish when satisfied with the results.""",
        "reference": "Reference",
        "reference_desc": "This project is inspired by research in AutoML systems. For a comprehensive overview of AutoML techniques, see:",
        "paper_title": "AutoML: A Survey of the State-of-the-Art",
        "view_paper": "View Paper (arXiv)",
        "paper_desc": """The paper covers neural architecture search, hyperparameter optimization, and meta-learning approaches
that form the theoretical foundation for modern AutoML systems.""",

        # Agents
        "data_agent": "Data Agent",
        "data_agent_role": "Data Analysis & Preprocessing",
        "data_agent_desc": "Analyzes your dataset to understand its structure, detects data types, identifies missing values, and proposes appropriate preprocessing steps (encoding, scaling, imputation).",
        "modeling_agent": "Modeling Agent",
        "modeling_agent_role": "Model Selection",
        "modeling_agent_desc": "Based on the data analysis, proposes candidate machine learning algorithms suitable for your task (classification or regression). Considers factors like dataset size, feature types, and complexity.",
        "hpo_agent": "HPO Agent",
        "hpo_agent_role": "Hyperparameter Optimization",
        "hpo_agent_desc": "Uses advanced optimization techniques (FLAML/Optuna) to find the best hyperparameters for each candidate model. Balances exploration vs exploitation to efficiently search the parameter space.",
        "eval_agent": "Evaluation Agent",
        "eval_agent_role": "Model Evaluation",
        "eval_agent_desc": "Evaluates trained models using appropriate metrics (accuracy, F1, RMSE, etc.), performs cross-validation, and compares models to identify the best performer.",
        "planner_agent": "Planner Agent",
        "planner_agent_role": "Orchestration (LLM)",
        "planner_agent_desc": "The 'brain' of the system. Uses GPT-4 to analyze the current state, decide the next action, and determine when to stop. Ensures agents execute in the correct order.",

        # Footer
        "footer_subtitle": "Multi-Agent Automated Machine Learning System",
        "footer_built": "Built by Juan Quintanilla | Powered by GPT-4, FLAML, and Streamlit",

        # Model explanations
        "lightgbm_name": "LightGBM",
        "lightgbm_desc": "Light Gradient Boosting Machine - A fast, distributed, high-performance gradient boosting framework based on decision trees. Excellent for large datasets and categorical features.",
        "xgboost_name": "XGBoost",
        "xgboost_desc": "Extreme Gradient Boosting - An optimized gradient boosting library designed for speed and performance. Industry standard for structured/tabular data.",
        "catboost_name": "CatBoost",
        "catboost_desc": "Categorical Boosting - A gradient boosting library that handles categorical features automatically using ordered boosting.",
        "random_forest_name": "Random Forest",
        "random_forest_desc": "An ensemble of decision trees that combines multiple trees to improve prediction accuracy and control overfitting.",
        "extra_trees_name": "Extra Trees",
        "extra_trees_desc": "Extremely Randomized Trees - Similar to Random Forest but with more randomization in tree building, often faster to train.",
    },
    "es": {
        # Navigation
        "nav_new_job": "Nuevo Trabajo",
        "nav_monitor": "Monitor",
        "nav_results": "Resultados",
        "nav_about": "Acerca de",
        "navigation": "Navegacion",
        "language": "Idioma",
        "quick_info": "Info Rapida",
        "quick_info_text": """**AutoML Agent** usa orquestacion
basada en LLM para automaticamente:
- Analizar tus datos
- Seleccionar mejores modelos
- Optimizar hiperparametros
- Evaluar rendimiento""",

        # Main
        "title": "AutoML Agent",
        "subtitle": "Machine Learning Automatizado con Orquestacion Multi-Agente",

        # New Job
        "create_new_job": "Crear Nuevo Trabajo AutoML",
        "upload_dataset": "Sube tu Dataset",
        "drop_csv": "Arrastra tu archivo CSV aqui",
        "or_use_sample": "O Usa un Dataset de Ejemplo",
        "test_system": "Prueba el sistema con datasets preconfigurados:",
        "select_sample": "-- Seleccionar --",
        "target": "Objetivo",
        "dataset_preview": "Vista Previa del Dataset",
        "rows": "Filas",
        "columns": "Columnas",
        "numeric": "Numericas",
        "missing_values": "Valores Faltantes",
        "target_column": "Columna Objetivo (variable a predecir)",
        "task_type": "Tipo de Tarea",
        "classification": "Clasificacion",
        "regression": "Regresion",
        "advanced_options": "Opciones Avanzadas",
        "config_path": "Ruta de Configuracion",
        "start_job": "Iniciar Trabajo AutoML",
        "creating_job": "Creando trabajo...",
        "job_created": "Trabajo creado exitosamente!",
        "job_id": "ID del Trabajo",
        "go_to_monitor": "Ve a **Monitor** para seguir el progreso",
        "api_error": "No se puede conectar a la API. Asegurate de que el servidor esta corriendo:",
        "dataset_not_found": "Dataset no encontrado",
        "error_loading": "Error cargando dataset",

        # Monitor
        "job_monitor": "Monitor de Trabajos",
        "no_jobs": "No hay trabajos. Crea uno nuevo para comenzar!",
        "total_jobs": "Total Trabajos",
        "running": "Ejecutando",
        "completed": "Completados",
        "failed": "Fallidos",
        "all_jobs": "Todos los Trabajos",
        "status": "Estado",
        "created": "Creado",
        "error": "Error",
        "job_details": "Detalles del Trabajo",
        "select_job": "Selecciona un trabajo para ver detalles",
        "duration": "Duracion",
        "iteration": "Iteracion",
        "view_logs": "Ver Logs",
        "no_logs": "No hay logs disponibles",
        "auto_refresh": "Actualizando automaticamente cada 5 segundos...",

        # Results
        "results": "Resultados",
        "no_completed": "No hay trabajos completados aun. Revisa el Monitor para trabajos en ejecucion.",
        "select_completed": "Seleccionar Trabajo Completado",
        "best_model": "Mejor Modelo",
        "algorithm": "Algoritmo",
        "training_time": "Tiempo de Entrenamiento",
        "model_details": "Detalles del Modelo",
        "key_strengths": "Fortalezas Clave",
        "auto_selected": "Seleccionado automaticamente como el modelo con mejor rendimiento para tu dataset.",
        "optimized_hyperparams": "Hiperparametros Optimizados",
        "parameter": "Parametro",
        "value": "Valor",
        "hyperparams_flaml": "Los hiperparametros fueron optimizados usando la busqueda AutoML de FLAML.",
        "hyperparams_auto": "Los hiperparametros fueron ajustados automaticamente. Revisa run_summary.json para detalles.",
        "all_metrics": "Todas las Metricas de Evaluacion",
        "download_model": "Descargar Modelo",
        "download_button": "Descargar Modelo Entrenado (.joblib)",
        "usage": "Uso",

        # About
        "about_title": "Acerca de AutoML Agent",
        "what_is_automl": "Que es AutoML?",
        "automl_desc": """Automated Machine Learning (AutoML) es el proceso de automatizar el proceso completo de aplicar
machine learning a problemas del mundo real. En lugar de seleccionar algoritmos manualmente, ajustar hiperparametros,
e ingenieria de features, los sistemas AutoML buscan automaticamente la mejor configuracion para tus datos.""",
        "multi_agent": "Arquitectura Multi-Agente",
        "multi_agent_desc": """Este sistema usa un **enfoque multi-agente** donde agentes de IA especializados colaboran para resolver el problema de machine learning.
Un LLM (Large Language Model) actua como orquestador, decidiendo que agente invocar segun el estado actual.""",
        "workflow": "Flujo de Trabajo",
        "workflow_desc": """El Agente Planificador orquesta todo el flujo de trabajo, asegurando que cada paso se complete antes de pasar al siguiente.
Puede decidir iterar (probar mas modelos o hiperparametros) o terminar cuando este satisfecho con los resultados.""",
        "reference": "Referencia",
        "reference_desc": "Este proyecto esta inspirado en investigacion de sistemas AutoML. Para una vision general de tecnicas AutoML, ver:",
        "paper_title": "AutoML: A Survey of the State-of-the-Art",
        "view_paper": "Ver Paper (arXiv)",
        "paper_desc": """El paper cubre busqueda de arquitecturas neuronales, optimizacion de hiperparametros y enfoques de meta-learning
que forman la base teorica de los sistemas AutoML modernos.""",

        # Agents
        "data_agent": "Agente de Datos",
        "data_agent_role": "Analisis de Datos y Preprocesamiento",
        "data_agent_desc": "Analiza tu dataset para entender su estructura, detecta tipos de datos, identifica valores faltantes, y propone pasos de preprocesamiento apropiados (codificacion, escalado, imputacion).",
        "modeling_agent": "Agente de Modelado",
        "modeling_agent_role": "Seleccion de Modelos",
        "modeling_agent_desc": "Basado en el analisis de datos, propone algoritmos de ML candidatos adecuados para tu tarea (clasificacion o regresion). Considera factores como tamano del dataset, tipos de features y complejidad.",
        "hpo_agent": "Agente HPO",
        "hpo_agent_role": "Optimizacion de Hiperparametros",
        "hpo_agent_desc": "Usa tecnicas de optimizacion avanzadas (FLAML/Optuna) para encontrar los mejores hiperparametros para cada modelo candidato. Balancea exploracion vs explotacion para buscar eficientemente en el espacio de parametros.",
        "eval_agent": "Agente de Evaluacion",
        "eval_agent_role": "Evaluacion de Modelos",
        "eval_agent_desc": "Evalua modelos entrenados usando metricas apropiadas (accuracy, F1, RMSE, etc.), realiza validacion cruzada, y compara modelos para identificar el mejor.",
        "planner_agent": "Agente Planificador",
        "planner_agent_role": "Orquestacion (LLM)",
        "planner_agent_desc": "El 'cerebro' del sistema. Usa GPT-4 para analizar el estado actual, decidir la siguiente accion, y determinar cuando parar. Asegura que los agentes se ejecuten en el orden correcto.",

        # Footer
        "footer_subtitle": "Sistema de Machine Learning Automatizado Multi-Agente",
        "footer_built": "Creado por Juan Quintanilla | Potenciado por GPT-4, FLAML y Streamlit",

        # Model explanations
        "lightgbm_name": "LightGBM",
        "lightgbm_desc": "Light Gradient Boosting Machine - Un framework de gradient boosting rapido, distribuido y de alto rendimiento basado en arboles de decision. Excelente para datasets grandes y features categoricas.",
        "xgboost_name": "XGBoost",
        "xgboost_desc": "Extreme Gradient Boosting - Una libreria de gradient boosting optimizada para velocidad y rendimiento. Estandar de la industria para datos estructurados/tabulares.",
        "catboost_name": "CatBoost",
        "catboost_desc": "Categorical Boosting - Una libreria de gradient boosting que maneja features categoricas automaticamente usando ordered boosting.",
        "random_forest_name": "Random Forest",
        "random_forest_desc": "Un ensemble de arboles de decision que combina multiples arboles para mejorar la precision de prediccion y controlar el overfitting.",
        "extra_trees_name": "Extra Trees",
        "extra_trees_desc": "Extremely Randomized Trees - Similar a Random Forest pero con mas aleatorizacion en la construccion de arboles, generalmente mas rapido de entrenar.",
    }
}

def get_text(key: str) -> str:
    """Get translated text for current language."""
    lang = st.session_state.get("language", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

# Sample datasets configuration
SAMPLE_DATASETS = {
    "Housing Prices (Regression)": {
        "path": "data/Housing.csv",
        "target": "price",
        "description": "Predict house prices based on features like area, bedrooms, location"
    },
    "Customer Churn (Classification)": {
        "path": "data/customer_churn.csv",
        "target": "churn",
        "description": "Predict if a customer will leave based on demographics and behavior"
    },
    "Titanic Survival (Classification)": {
        "path": "data/dataset.csv",
        "target": "Survived",
        "description": "Predict passenger survival based on class, age, gender"
    },
    "Life Expectancy (Regression)": {
        "path": "data/life-expectancy.csv",
        "target": "Life expectancy ",
        "description": "Predict life expectancy based on health and economic factors"
    },
    "Student Scores (Regression)": {
        "path": "data/student_scores.csv",
        "target": "Scores",
        "description": "Predict exam scores based on study hours"
    }
}

# Contact information
CONTACT_INFO = {
    "email": "juanquiber2003@gmail.com",
    "linkedin": "https://www.linkedin.com/in/juan-quintanilla-bernabe/",
    "github": "https://github.com/juanquintanilla-profile/",
    "paper": "https://arxiv.org/pdf/1908.00709"
}


def apply_custom_css():
    """Apply custom CSS for a cleaner, professional look."""
    st.markdown("""
    <style>
        /* Language toggle button - top right */
        .lang-toggle {
            position: fixed;
            top: 14px;
            right: 100px;
            z-index: 1000;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 6px 14px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            transition: all 0.2s ease;
        }

        .lang-toggle:hover {
            transform: scale(1.05);
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
            color: white;
        }

        /* Main container */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* Headers */
        h1 {
            color: #1a1a2e;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        h2, h3 {
            color: #16213e;
            font-weight: 600;
        }

        /* Cards */
        .info-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1rem;
        }

        .metric-card {
            background: #f8f9fa;
            padding: 1.2rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin-bottom: 0.8rem;
        }

        .metric-card h4 {
            margin: 0;
            color: #666;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .metric-card p {
            margin: 0.3rem 0 0 0;
            color: #1a1a2e;
            font-size: 1.4rem;
            font-weight: 600;
        }

        /* Dataset cards */
        .dataset-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            margin-bottom: 0.5rem;
            transition: all 0.2s ease;
        }

        .dataset-card:hover {
            border-color: #667eea;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
        }

        /* Footer */
        .footer {
            background: #1a1a2e;
            color: #a0a0a0;
            padding: 2rem;
            border-radius: 12px;
            margin-top: 3rem;
        }

        .footer a {
            color: #667eea;
            text-decoration: none;
        }

        .footer a:hover {
            color: #764ba2;
        }

        /* Status badges */
        .status-badge {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .status-pending { background: #fff3cd; color: #856404; }
        .status-running { background: #cce5ff; color: #004085; }
        .status-completed { background: #d4edda; color: #155724; }
        .status-failed { background: #f8d7da; color: #721c24; }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        /* Sidebar */
        .css-1d391kg {
            background: #f8f9fa;
        }

        /* Tables */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            font-weight: 500;
            color: #1a1a2e;
        }

        /* Agent info boxes */
        .agent-box {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 3px solid #667eea;
        }

        .agent-box h5 {
            margin: 0 0 0.5rem 0;
            color: #1a1a2e;
        }

        .agent-box p {
            margin: 0;
            color: #666;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)


st.set_page_config(
    page_title="AutoML Agent",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main Streamlit app."""
    # Initialize language in session state
    if "language" not in st.session_state:
        st.session_state.language = "en"

    apply_custom_css()

    # Language toggle button (top right) - using query params for state
    current_lang = st.session_state.language
    next_lang = "es" if current_lang == "en" else "en"
    lang_label = "ES" if current_lang == "en" else "EN"

    # Create a hidden button and use JavaScript to style it
    col_spacer, col_lang = st.columns([10, 1])
    with col_lang:
        if st.button(lang_label, key="lang_toggle", help=f"Switch to {'Spanish' if current_lang == 'en' else 'English'}"):
            st.session_state.language = next_lang
            st.rerun()

    # Header
    st.title(get_text("title"))
    st.markdown(f"**{get_text('subtitle')}**")

    # Sidebar navigation
    st.sidebar.markdown(f"### {get_text('navigation')}")
    page_options = [
        get_text("nav_new_job"),
        get_text("nav_monitor"),
        get_text("nav_results"),
        get_text("nav_about")
    ]
    page = st.sidebar.radio(
        "",
        page_options,
        label_visibility="collapsed"
    )

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {get_text('quick_info')}")
    st.sidebar.markdown(get_text("quick_info_text"))

    if page == get_text("nav_new_job"):
        show_new_job_page()
    elif page == get_text("nav_monitor"):
        show_monitor_page()
    elif page == get_text("nav_results"):
        show_results_page()
    elif page == get_text("nav_about"):
        show_about_page()

    # Footer on all pages
    show_footer()


def show_new_job_page():
    """Page for creating new AutoML jobs."""
    st.header(get_text("create_new_job"))

    # Upload section
    st.subheader(get_text("upload_dataset"))
    uploaded_file = st.file_uploader(
        get_text("drop_csv"),
        type=["csv"],
        help="Upload a CSV file with your training data"
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        show_dataset_config(df, uploaded_file, is_sample=False)
    else:
        # Only show sample datasets if no file uploaded
        st.markdown("---")
        st.subheader(get_text("or_use_sample"))
        st.markdown(get_text("test_system"))

        selected_sample = st.selectbox(
            "Select a sample dataset",
            options=[get_text("select_sample")] + list(SAMPLE_DATASETS.keys()),
            label_visibility="collapsed"
        )

        if selected_sample != get_text("select_sample"):
            sample_info = SAMPLE_DATASETS[selected_sample]

            st.markdown(f"""
            <div class="dataset-card">
                <strong>{selected_sample}</strong><br>
                <small style="color: #666;">{sample_info['description']}</small><br>
                <small><strong>{get_text("target")}:</strong> {sample_info['target']}</small>
            </div>
            """, unsafe_allow_html=True)

            # Load and show sample dataset
            try:
                sample_path = Path(sample_info['path'])
                if sample_path.exists():
                    df = pd.read_csv(sample_path)
                    show_dataset_config(
                        df,
                        sample_path,
                        is_sample=True,
                        default_target=sample_info['target']
                    )
                else:
                    st.error(f"{get_text('dataset_not_found')}: {sample_info['path']}")
            except Exception as e:
                st.error(f"{get_text('error_loading')}: {str(e)}")


def show_dataset_config(df: pd.DataFrame, source, is_sample: bool = False, default_target: str = None):
    """Show dataset configuration and submit button."""

    st.markdown("---")
    st.subheader(get_text("dataset_preview"))
    st.dataframe(df.head(8), use_container_width=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>{get_text("rows")}</h4>
            <p>{len(df):,}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>{get_text("columns")}</h4>
            <p>{len(df.columns)}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        numeric_cols = df.select_dtypes(include=['number']).columns
        st.markdown(f"""
        <div class="metric-card">
            <h4>{get_text("numeric")}</h4>
            <p>{len(numeric_cols)}</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        missing = df.isnull().sum().sum()
        st.markdown(f"""
        <div class="metric-card">
            <h4>{get_text("missing_values")}</h4>
            <p>{missing:,}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Target selection
    col1, col2 = st.columns([2, 1])

    with col1:
        default_idx = 0
        if default_target and default_target in df.columns.tolist():
            default_idx = df.columns.tolist().index(default_target)

        target_column = st.selectbox(
            get_text("target_column"),
            options=df.columns.tolist(),
            index=default_idx,
            help="Select the column you want to predict"
        )

    with col2:
        # Detect task type
        if target_column:
            unique_values = df[target_column].nunique()
            if unique_values <= 10:
                task_type = get_text("classification")
            else:
                task_type = get_text("regression")

            st.markdown(f"""
            <div class="metric-card">
                <h4>{get_text("task_type")}</h4>
                <p>{task_type}</p>
            </div>
            """, unsafe_allow_html=True)

    # Advanced options
    with st.expander(get_text("advanced_options")):
        config_path = st.text_input(
            get_text("config_path"),
            value="automl_agent/config.yaml",
            help="Path to custom configuration file"
        )

    # Submit button
    st.markdown("")
    if st.button(get_text("start_job"), type="primary", use_container_width=True):
        run_automl_job(source, target_column, config_path if 'config_path' in dir() else "automl_agent/config.yaml", is_sample)


def run_automl_job(source, target_column: str, config_path: str, is_sample: bool):
    """Submit the AutoML job to the API."""
    with st.spinner(get_text("creating_job")):
        try:
            if is_sample:
                # Read sample file
                with open(source, 'rb') as f:
                    response = requests.post(
                        f"{API_URL}/api/v1/jobs",
                        files={"dataset": (source.name, f, "text/csv")},
                        data={
                            "target_column": target_column,
                            "config_path": config_path,
                        }
                    )
            else:
                # Uploaded file
                source.seek(0)
                response = requests.post(
                    f"{API_URL}/api/v1/jobs",
                    files={"dataset": source},
                    data={
                        "target_column": target_column,
                        "config_path": config_path,
                    }
                )

            if response.status_code == 200:
                job_data = response.json()
                st.success(get_text("job_created"))
                st.info(f"**{get_text('job_id')}:** `{job_data['job_id']}`")
                st.info(get_text("go_to_monitor"))

                # Store job ID in session state
                if 'job_ids' not in st.session_state:
                    st.session_state.job_ids = []
                st.session_state.job_ids.append(job_data['job_id'])
            else:
                st.error(f"Error: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(get_text("api_error"))
            st.code("python -m uvicorn api.main:app --reload")
        except Exception as e:
            st.error(f"Error: {str(e)}")


def show_monitor_page():
    """Page for monitoring job progress."""
    st.header(get_text("job_monitor"))

    try:
        response = requests.get(f"{API_URL}/api/v1/jobs")

        if response.status_code == 200:
            data = response.json()
            jobs = data["jobs"]

            if not jobs:
                st.info(get_text("no_jobs"))
                return

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            running = len([j for j in jobs if j["status"] == "running"])
            completed = len([j for j in jobs if j["status"] == "completed"])
            failed = len([j for j in jobs if j["status"] == "failed"])
            pending = len([j for j in jobs if j["status"] == "pending"])

            with col1:
                st.metric(get_text("total_jobs"), data['total'])
            with col2:
                st.metric(get_text("running"), running)
            with col3:
                st.metric(get_text("completed"), completed)
            with col4:
                st.metric(get_text("failed"), failed)

            st.markdown("---")

            # Jobs table
            st.subheader(get_text("all_jobs"))

            # Build table data
            table_data = []
            for job in jobs:
                status = job["status"].upper()
                table_data.append({
                    get_text("job_id"): job["job_id"][:12] + "...",
                    get_text("status"): status,
                    get_text("target"): job["target_column"],
                    get_text("created"): datetime.fromisoformat(job["created_at"]).strftime("%Y-%m-%d %H:%M"),
                    get_text("error"): (job.get("error", "")[:40] + "...") if job.get("error") and len(job.get("error", "")) > 40 else job.get("error", "")
                })

            jobs_df = pd.DataFrame(table_data)
            st.dataframe(jobs_df, use_container_width=True, hide_index=True)

            # Job detail section
            st.markdown("---")
            st.subheader(get_text("job_details"))

            selected_job = st.selectbox(
                get_text("select_job"),
                options=[job["job_id"] for job in jobs],
                format_func=lambda x: f"{x[:12]}... ({next(j['status'].upper() for j in jobs if j['job_id'] == x)})"
            )

            if selected_job:
                show_job_details(selected_job)

        else:
            st.error(f"Error fetching jobs: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error(get_text("api_error"))
    except Exception as e:
        st.error(f"Error: {str(e)}")


def show_job_details(job_id: str):
    """Show detailed information for a specific job."""
    try:
        response = requests.get(f"{API_URL}/api/v1/jobs/{job_id}/status")

        if response.status_code == 200:
            job = response.json()

            col1, col2, col3 = st.columns(3)

            with col1:
                status_class = f"status-{job['status']}"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{get_text("status")}</h4>
                    <p><span class="status-badge {status_class}">{job['status'].upper()}</span></p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                duration = job.get("duration_seconds", 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{get_text("duration")}</h4>
                    <p>{duration:.1f}s</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                iteration = job.get("current_iteration", "-")
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{get_text("iteration")}</h4>
                    <p>{iteration}</p>
                </div>
                """, unsafe_allow_html=True)

            if job.get("error"):
                st.error(f"**Error:** {job['error']}")

            # Logs
            with st.expander(get_text("view_logs")):
                logs_response = requests.get(f"{API_URL}/api/v1/jobs/{job_id}/logs")
                if logs_response.status_code == 200:
                    logs = logs_response.json()
                    st.code(logs.get("logs", get_text("no_logs")), language="text")

            # Auto-refresh for running jobs
            if job["status"] == "running":
                st.info(get_text("auto_refresh"))
                time.sleep(5)
                st.rerun()

        else:
            st.error(f"Error fetching job details: {response.text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")


def show_results_page():
    """Page for viewing job results."""
    st.header(get_text("results"))

    try:
        response = requests.get(f"{API_URL}/api/v1/jobs")

        if response.status_code == 200:
            data = response.json()
            completed_jobs = [
                job for job in data["jobs"]
                if job["status"] == "completed"
            ]

            if not completed_jobs:
                st.info(get_text("no_completed"))
                return

            selected_job = st.selectbox(
                get_text("select_completed"),
                options=[job["job_id"] for job in completed_jobs],
                format_func=lambda x: f"{x[:12]}... ({next(j['target_column'] for j in completed_jobs if j['job_id'] == x)})"
            )

            if selected_job:
                show_job_results(selected_job)

        else:
            st.error(f"Error fetching jobs: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error(get_text("api_error"))
    except Exception as e:
        st.error(f"Error: {str(e)}")


def show_job_results(job_id: str):
    """Display results for a completed job."""
    try:
        response = requests.get(f"{API_URL}/api/v1/jobs/{job_id}/results")

        if response.status_code == 200:
            results = response.json()

            st.markdown("---")

            # Best Model Section
            st.subheader(get_text("best_model"))

            col1, col2, col3 = st.columns(3)

            with col1:
                model_name = results.get("best_model", "N/A")
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{get_text("algorithm")}</h4>
                    <p>{model_name}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if results.get("best_score"):
                    metric_name = results.get("primary_metric", "Score")
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{metric_name}</h4>
                        <p>{results['best_score']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col3:
                if results.get("total_time"):
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{get_text("training_time")}</h4>
                        <p>{results['total_time']:.1f}s</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Model Explanation
            st.markdown("---")
            st.subheader(get_text("model_details"))

            model_key = model_name.lower().replace("-", "_").replace(" ", "_") if model_name else ""

            model_keys = ["lightgbm", "xgboost", "catboost", "random_forest", "extra_trees"]
            model_strengths = {
                "en": {
                    "lightgbm": ["Fast training speed", "Low memory usage", "High accuracy", "Handles categorical features natively"],
                    "xgboost": ["Regularization to prevent overfitting", "Parallel processing", "Handles missing values", "Feature importance"],
                    "catboost": ["Best for categorical data", "Reduces overfitting", "No preprocessing needed", "GPU support"],
                    "random_forest": ["Robust to outliers", "Feature importance", "Less prone to overfitting", "Works well with mixed data types"],
                    "extra_trees": ["Faster training", "More randomization", "Good generalization", "Handles high-dimensional data"]
                },
                "es": {
                    "lightgbm": ["Entrenamiento rapido", "Bajo uso de memoria", "Alta precision", "Maneja features categoricas nativamente"],
                    "xgboost": ["Regularizacion para prevenir overfitting", "Procesamiento paralelo", "Maneja valores faltantes", "Importancia de features"],
                    "catboost": ["Ideal para datos categoricos", "Reduce overfitting", "No necesita preprocesamiento", "Soporte GPU"],
                    "random_forest": ["Robusto ante outliers", "Importancia de features", "Menos propenso a overfitting", "Funciona bien con datos mixtos"],
                    "extra_trees": ["Entrenamiento mas rapido", "Mas aleatorizacion", "Buena generalizacion", "Maneja datos de alta dimension"]
                }
            }

            if model_key in model_keys:
                st.markdown(f"**{get_text(f'{model_key}_name')}**")
                st.markdown(get_text(f"{model_key}_desc"))

                lang = st.session_state.get("language", "en")
                st.markdown(f"**{get_text('key_strengths')}:**")
                for strength in model_strengths[lang].get(model_key, []):
                    st.markdown(f"- {strength}")
            else:
                st.markdown(f"**{model_name}** - {get_text('auto_selected')}")

            # Hyperparameters
            st.markdown("---")
            st.subheader(get_text("optimized_hyperparams"))

            if results.get("best_params") or results.get("hyperparameters"):
                params = results.get("best_params") or results.get("hyperparameters", {})
                if params:
                    # Display as a nice table
                    params_df = pd.DataFrame([
                        {get_text("parameter"): k, get_text("value"): str(v)}
                        for k, v in params.items()
                    ])
                    st.dataframe(params_df, use_container_width=True, hide_index=True)
                else:
                    st.info(get_text("hyperparams_flaml"))
            else:
                st.info(get_text("hyperparams_auto"))

            # All Metrics
            if results.get("all_metrics"):
                st.markdown("---")
                st.subheader(get_text("all_metrics"))

                metrics = results["all_metrics"]

                # Display metrics in a grid
                metric_cols = st.columns(min(len(metrics), 4))
                for i, (metric_name, metric_value) in enumerate(metrics.items()):
                    with metric_cols[i % 4]:
                        if isinstance(metric_value, float):
                            st.metric(metric_name, f"{metric_value:.4f}")
                        else:
                            st.metric(metric_name, str(metric_value))

            # Download Section
            st.markdown("---")
            st.subheader(get_text("download_model"))

            col1, col2 = st.columns([1, 2])

            with col1:
                download_url = f"{API_URL}/api/v1/jobs/{job_id}/download-model"

                try:
                    model_response = requests.get(download_url)
                    if model_response.status_code == 200:
                        st.download_button(
                            label=get_text("download_button"),
                            data=model_response.content,
                            file_name=f"model_{job_id[:8]}.joblib",
                            mime="application/octet-stream",
                            type="primary"
                        )
                except:
                    st.markdown(f"[Download Model]({download_url})")

            with col2:
                st.markdown(f"""
                **{get_text("usage")}:**
                ```python
                import joblib
                model = joblib.load('model.joblib')
                predictions = model.predict(X_new)
                ```
                """)

        else:
            st.error(f"Error fetching results: {response.text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")


def show_about_page():
    """About page with AutoML explanation."""
    st.header(get_text("about_title"))

    # What is AutoML
    st.markdown(f"""
    <div class="info-card">
        <h3 style="margin-top:0; color:white;">{get_text("what_is_automl")}</h3>
        <p style="margin-bottom:0;">
            {get_text("automl_desc")}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Multi-Agent Architecture
    st.subheader(get_text("multi_agent"))

    st.markdown(get_text("multi_agent_desc"))

    # Agent descriptions
    agents = [
        {"name": get_text("data_agent"), "role": get_text("data_agent_role"), "description": get_text("data_agent_desc")},
        {"name": get_text("modeling_agent"), "role": get_text("modeling_agent_role"), "description": get_text("modeling_agent_desc")},
        {"name": get_text("hpo_agent"), "role": get_text("hpo_agent_role"), "description": get_text("hpo_agent_desc")},
        {"name": get_text("eval_agent"), "role": get_text("eval_agent_role"), "description": get_text("eval_agent_desc")},
        {"name": get_text("planner_agent"), "role": get_text("planner_agent_role"), "description": get_text("planner_agent_desc")},
    ]

    for agent in agents:
        st.markdown(f"""
        <div class="agent-box">
            <h5>{agent['name']} - {agent['role']}</h5>
            <p>{agent['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Workflow diagram
    st.subheader(get_text("workflow"))

    st.markdown("""
    ```
    Data Agent  -->  Modeling Agent  -->  HPO Agent  -->  Evaluation Agent
         |                                                        |
         |                                                        v
         +------------------  Planner Agent  <--------------------+
                                   |
                          (iterate or finish)
    ```
    """)

    st.markdown(get_text("workflow_desc"))

    st.markdown("---")

    # Reference paper
    st.subheader(get_text("reference"))

    st.markdown(f"""
    {get_text("reference_desc")}

    **{get_text("paper_title")}**
    [{get_text("view_paper")}]({CONTACT_INFO['paper']})

    {get_text("paper_desc")}
    """)


def show_footer():
    """Display footer with contact information."""
    st.markdown("---")

    st.markdown(f"""
    <div class="footer">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <strong style="color: white;">AutoML Agent</strong><br>
                <small>{get_text("footer_subtitle")}</small>
            </div>
            <div style="text-align: right;">
                <a href="mailto:{CONTACT_INFO['email']}" title="Email">Email</a> |
                <a href="{CONTACT_INFO['linkedin']}" target="_blank" title="LinkedIn">LinkedIn</a> |
                <a href="{CONTACT_INFO['github']}" target="_blank" title="GitHub">GitHub</a> |
                <a href="{CONTACT_INFO['paper']}" target="_blank" title="Reference Paper">Paper</a>
            </div>
        </div>
        <div style="margin-top: 1rem; text-align: center;">
            <small>{get_text("footer_built")}</small>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
