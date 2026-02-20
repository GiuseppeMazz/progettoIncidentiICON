from sklearn.model_selection import KFold
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.base import clone





def k_fold_personalizzato(modello, X, y, nome):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_y_test = []
    all_y_pred = []
    fold_mae = []
    fold_mse = []

    for train_index, test_index in kf.split(X):

        model = clone(modello)

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_y_test.append(y_test)
        all_y_pred.append(y_pred)

        fold_mae.append(mean_absolute_error(y_test, y_pred))
        fold_mse.append(mean_squared_error(y_test, y_pred))

    fold_mae = np.array(fold_mae)
    fold_mse = np.array(fold_mse)

    median_idx = np.argsort(fold_mae)[len(fold_mae)//2]

    risultati = {
        "nome": nome,
        "test_mediano": all_y_test[median_idx],
        "pred_mediano": all_y_pred[median_idx],
        "mae_mean": fold_mae.mean(),
        "mae_std": fold_mae.std(),
        "best_mae": fold_mae.min(),
        "worst_mae": fold_mae.max(),
        "mse_mean": fold_mse.mean(),
        "mse_std": fold_mse.std(),
        "best_mse": fold_mse.min(),
        "worst_mse": fold_mse.max()
    }

    return risultati




def valuta_modello_regressione(y_true, y_pred, titolo_modello="Modello di Regressione"):
    #Visualizza grafici e metriche per valutare un modello di regressione.

    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Calcolo metriche
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Crea la figura
    fig=plt.figure(figsize=(14, 6))

    # Reale vs Predetto
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor=None)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label="Ideale")
    plt.xlabel("Valori Reali")
    plt.ylabel("Valori Predetti")
    plt.title(f"{titolo_modello}\nPredetto vs Reale")
    plt.legend()

    # Residual plot: Errori vs Predetto
    residui = y_true - y_pred
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_pred, y=residui, alpha=0.6, edgecolor=None)


    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Valori Predetti")
    plt.ylabel("Residui (Reale - Predetto)")
    plt.title("Distribuzione dei Residui")

    # Layout
    plt.suptitle(f"Valutazione {titolo_modello}\n"
                 f"MAE={mae:.3f} | RMSE={rmse:.3f} | R²={r2:.3f}",
                 fontsize=14, y=1.03)

    plt.tight_layout()
    #plt.show()
    return fig




def ottieni_prestazioni_modelli(lista_risultati):
    for risultati in lista_risultati:
        print(f"\n=== RISULTATI MODELLO {risultati["nome"]} ===")

        print(f"{risultati["nome"]} MAE medio: {risultati['mae_mean']:.4f}, MAE variabilità: {risultati['mae_std']:.4f}")
        print(f"{risultati["nome"]} miglior MAE: {risultati['best_mae']:.4f}, peggior MAE: {risultati['worst_mae']:.4f}")

        print(f"{risultati["nome"]} MSE medio: {risultati['mse_mean']:.4f}, MSE variabilità: {risultati['mse_std']:.4f}")
        print(f"{risultati["nome"]} miglior MSE: {risultati['best_mse']:.4f}, peggior MSE: {risultati['worst_mse']:.4f}")
        #print("\n")
        #produci_output_modelli(risultati, risultati["nome"])



#inserisci grafici

def ottieni_grafici(lista_risultati):
    for risultati in lista_risultati:
        grafico = valuta_modello_regressione(risultati['test_mediano'], risultati['pred_mediano'], risultati["nome"])


        os.makedirs("grafici", exist_ok=True)
        grafico.savefig(f"grafici/{risultati["nome"]}.png", dpi=300, bbox_inches="tight")
        print(f"Grafico {risultati["nome"]} salvato nella cartella 'grafici'")
        plt.close(grafico)


#grafico per la valutazione della feature selection sequenziale
def plot_metric(n_features_range, mae_scores, mse_scores, nome):
    # ---- Grafico MAE ----
    fig_mae, ax_mae = plt.subplots()
    ax_mae.plot(n_features_range, mae_scores, marker='o')
    ax_mae.set_xlabel("Numero di feature")
    ax_mae.set_ylabel("MAE")
    ax_mae.set_title(f"MAE vs Numero di feature - {nome}")
    ax_mae.grid(True)

    # ---- Grafico MSE ----
    fig_mse, ax_mse = plt.subplots()
    ax_mse.plot(n_features_range, mse_scores, marker='o')
    ax_mse.set_xlabel("Numero di feature")
    ax_mse.set_ylabel("MSE")
    ax_mse.set_title(f"MSE vs Numero di feature - {nome}")
    ax_mse.grid(True)

    return fig_mae, fig_mse

def ottieni_grafici_features_selection(n_features_range, mae_scores, mse_scores, nomeImg):
    os.makedirs("grafici", exist_ok=True)
    grafico_mae, grafico_mse = plot_metric(n_features_range, mae_scores, mse_scores, nomeImg)

    grafico_mae.savefig(f"grafici/{nomeImg}_mae.png", dpi=300, bbox_inches="tight")
    print(f"Grafico {nomeImg}_mae salvato nella cartella 'grafici'")
    plt.close(grafico_mae)
    grafico_mse.savefig(f"grafici/{nomeImg}_mse.png", dpi=300, bbox_inches="tight")
    print(f"Grafico {nomeImg}_mse salvato nella cartella 'grafici'")
    plt.close(grafico_mse)
    plt.close('all')

