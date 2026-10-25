# Path sirve para trabajar con rutas de archivos y carpetas
# de forma independiente del sistema operativo.
# Es preferible a concatenar strings manualmente.
from pathlib import Path

# NumPy lo usaremos para cálculos numéricos y,
# en este caso, para generar variables aleatorias.
import numpy as np

# Pandas lo usaremos para trabajar con datos tabulares,
# equivalente conceptualmente a data.frame / tibble en R.
import pandas as pd


def generate_customer_data(
    n_customers: int = 5_000,
    random_state: int = 123,
) -> pd.DataFrame:
    """
    Genera un dataset sintético para un modelo de propensión.

    Parámetros
    ----------
    n_customers:
        Número de clientes que queremos generar.
        Por defecto, 5.000.

    random_state:
        Semilla del generador aleatorio.
        Permite que los resultados sean reproducibles.

    Devuelve
    --------
    pd.DataFrame
        Un DataFrame de pandas con variables de cliente
        y una variable objetivo binaria llamada 'target'.
    """

    # Creamos un generador de números aleatorios.
    #
    # Usar una semilla fija hace que podamos volver a generar
    # exactamente el mismo dataset.
    #
    # Conceptualmente sería parecido a set.seed(123) en R.
    rng = np.random.default_rng(random_state)

    # ---------------------------------------------------------
    # GENERACIÓN DE VARIABLES PREDICTORAS
    # ---------------------------------------------------------

    # Edad del cliente.
    #
    # Generamos enteros aleatorios desde 18 hasta 79.
    # El límite superior 80 no se incluye.
    #
    # size=n_customers significa que generamos un valor
    # por cada cliente.
    age = rng.integers(
        18,
        80,
        size=n_customers,
    )

    # Antigüedad del cliente en meses.
    #
    # Generamos valores entre 1 y 120 meses.
    tenure_months = rng.integers(
        1,
        121,
        size=n_customers,
    )

    # Gasto mensual.
    #
    # Usamos una distribución Gamma porque produce
    # valores positivos y con asimetría hacia la derecha,
    # algo razonable para una variable de gasto.
    monthly_spend = rng.gamma(
        shape=2.5,
        scale=35,
        size=n_customers,
    )

    # Número de compras anteriores.
    #
    # Una distribución Poisson resulta útil para modelizar
    # variables de conteo.
    #
    # lam=3 indica una media aproximada de 3 compras.
    previous_purchases = rng.poisson(
        lam=3,
        size=n_customers,
    )

    # Número de contactos de campañas anteriores.
    #
    # También es una variable de conteo.
    # En este caso usamos una Poisson con media 2.
    campaign_contacts = rng.poisson(
        lam=2,
        size=n_customers,
    )

    # Indicador de cliente digital.
    #
    # Generamos una variable binaria:
    #
    # 1 = cliente digital
    # 0 = cliente no digital
    #
    # La probabilidad de ser digital es 0.65.
    digital_customer = rng.binomial(
        n=1,
        p=0.65,
        size=n_customers,
    )

    # ---------------------------------------------------------
    # CONSTRUCCIÓN DE LA PROPENSIÓN REAL
    # ---------------------------------------------------------

    # Creamos un predictor lineal.
    #
    # Es equivalente a la parte lineal de una regresión logística:
    #
    # eta =
    # beta0 +
    # beta1 * X1 +
    # beta2 * X2 +
    # ...
    #
    # Los coeficientes son inventados, pero nos permiten simular
    # un problema realista en el que ciertas variables aumentan
    # o disminuyen la propensión.
    linear_predictor = (
        -3.0

        # Una mayor antigüedad aumenta ligeramente
        # la probabilidad de responder.
        + 0.012 * tenure_months

        # Un mayor gasto mensual también incrementa
        # la probabilidad de respuesta.
        + 0.009 * monthly_spend

        # Clientes con más compras anteriores
        # tienen mayor propensión.
        + 0.20 * previous_purchases

        # Más contactos comerciales previos reducen
        # la probabilidad de respuesta.
        #
        # Puede interpretarse, por ejemplo, como fatiga comercial.
        - 0.15 * campaign_contacts

        # Los clientes digitales tienen una propensión mayor.
        + 0.55 * digital_customer

        # La edad tiene en este ejemplo
        # un pequeño efecto negativo.
        - 0.006 * age
    )

    # ---------------------------------------------------------
    # TRANSFORMACIÓN LOGÍSTICA
    # ---------------------------------------------------------

    # El predictor lineal puede tomar cualquier valor
    # entre -infinito y +infinito.
    #
    # Para convertirlo en una probabilidad entre 0 y 1
    # aplicamos la función logística:
    #
    # p = 1 / (1 + exp(-eta))
    #
    # Esta es exactamente la transformación utilizada
    # por una regresión logística.
    propensity = 1 / (1 + np.exp(-linear_predictor))

    # ---------------------------------------------------------
    # GENERACIÓN DEL TARGET
    # ---------------------------------------------------------

    # Ahora generamos el resultado observado.
    #
    # Cada cliente tiene una probabilidad distinta
    # de responder, dada por 'propensity'.
    #
    # rng.binomial(1, propensity) genera:
    #
    # 1 = responde
    # 0 = no responde
    #
    # con la probabilidad individual correspondiente.
    target = rng.binomial(
        n=1,
        p=propensity,
    )

    # ---------------------------------------------------------
    # CREACIÓN DEL DATAFRAME FINAL
    # ---------------------------------------------------------

    # Construimos un DataFrame de pandas.
    #
    # Cada clave del diccionario será una columna.
    # Cada array contiene los valores de esa columna.
    return pd.DataFrame(
        {
            # Identificador único del cliente.
            #
            # np.arange(1, n_customers + 1) genera:
            #
            # 1, 2, 3, ..., n_customers
            "customer_id": np.arange(
                1,
                n_customers + 1,
            ),

            "age": age,

            "tenure_months": tenure_months,

            # Redondeamos el gasto a dos decimales
            # para que parezca una cantidad monetaria real.
            "monthly_spend": monthly_spend.round(2),

            "previous_purchases": previous_purchases,

            "campaign_contacts": campaign_contacts,

            "digital_customer": digital_customer,

            # Variable objetivo del modelo.
            "target": target,
        }
    )


def save_data(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Guarda un DataFrame en formato CSV.

    Parámetros
    ----------
    data:
        DataFrame que queremos guardar.

    output_path:
        Ruta en la que se escribirá el archivo CSV.

    Devuelve
    --------
    None
        La función no devuelve ningún objeto.
        Su efecto es crear un archivo en disco.
    """

    # Creamos la carpeta padre si todavía no existe.
    #
    # parents=True:
    # crea también carpetas intermedias si son necesarias.
    #
    # exist_ok=True:
    # no genera error si la carpeta ya existe.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Guardamos el DataFrame como CSV.
    #
    # index=False evita que pandas escriba
    # una columna adicional con el índice del DataFrame.
    data.to_csv(
        output_path,
        index=False,
    )


# -------------------------------------------------------------
# PUNTO DE ENTRADA DEL SCRIPT
# -------------------------------------------------------------

# Este bloque solamente se ejecuta cuando hacemos algo como:
#
# uv run python -m propensity_model_python.data
#
# Si el fichero se importa desde otro módulo, este bloque
# NO se ejecuta.
if __name__ == "__main__":

    # Generamos el dataset utilizando
    # los valores por defecto:
    #
    # n_customers = 5000
    # random_state = 123
    dataset = generate_customer_data()

    # Definimos dónde queremos guardar el CSV.
    #
    # Path crea un objeto de tipo ruta.
    output_path = Path(
        "data/raw/customers.csv"
    )

    # Guardamos el dataset.
    save_data(
        dataset,
        output_path,
    )

    # Mostramos cuántos clientes hemos generado.
    #
    # :, dentro del f-string añade separador de miles.
    #
    # Por ejemplo:
    # 5000 -> 5,000
    print(
        f"Generated {len(dataset):,} customers"
    )

    # Calculamos la media del target.
    #
    # Como target solo contiene 0 y 1:
    #
    # mean(target)
    #
    # equivale a la proporción de respuestas positivas.
    #
    # :.3f significa mostrar tres decimales.
    print(
        f"Response rate: "
        f"{dataset['target'].mean():.3f}"
    )