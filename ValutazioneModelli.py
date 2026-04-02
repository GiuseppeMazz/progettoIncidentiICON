from sklearn.model_selection import KFold
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV


def k_fold_personalizzato(modello, X, y, nome):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_y_test = []
    all_y_pred = []
    fold_mae = []
    fold_mse = []
    fold_r2 = []  # aggiunto

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
        fold_r2.append(r2_score(y_test, y_pred))  # aggiunto

    fold_mae = np.array(fold_mae)
    fold_mse = np.array(fold_mse)
    fold_r2  = np.array(fold_r2)   # aggiunto

    median_idx = np.argsort(fold_mae)[len(fold_mae) // 2]

    risultati = {
        "nome": nome,
        "test_mediano": all_y_test[median_idx],
        "pred_mediano": all_y_pred[median_idx],
        "mae_mean":  fold_mae.mean(),
        "mae_std":   fold_mae.std(),
        "best_mae":  fold_mae.min(),
        "worst_mae": fold_mae.max(),
        "mse_mean":  fold_mse.mean(),
        "mse_std":   fold_mse.std(),
        "best_mse":  fold_mse.min(),
        "worst_mse": fold_mse.max(),
        "r2_mean":   fold_r2.mean(),   # aggiunto
        "r2_std":    fold_r2.std(),    # aggiunto
        "best_r2":   fold_r2.max(),    # aggiunto (max perché R² alto è meglio)
        "worst_r2":  fold_r2.min(),    # aggiunto
    }

    return risultati

#KFOLD personalizzato per approccio  con gridsearch

def nested_kfold_grid_src(modello, param_grid, X, y, nome):

    outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_mae = []
    fold_mse = []
    fold_r2 = []

    for train_index, test_index in outer_kf.split(X):
        print(f"  Fold esterno {nome} iniziato...", flush=True)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # inner CV per hyperparameter tuning
        grid = GridSearchCV(
            clone(modello),
            param_grid,
            cv=5,
            scoring="neg_mean_absolute_error"
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)

        fold_mae.append(mean_absolute_error(y_test, y_pred))
        fold_mse.append(mean_squared_error(y_test, y_pred))
        fold_r2.append(r2_score(y_test, y_pred))

    risultati = {
        "nome": nome,
        "mae_mean": np.mean(fold_mae),
        "mae_std": np.std(fold_mae),
        "mse_mean": np.mean(fold_mse),
        "mse_std": np.std(fold_mse),
        "r2_mean": np.mean(fold_r2),
        "r2_std": np.std(fold_r2)
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
        print(f"\n=== RISULTATI MODELLO {risultati['nome']} ===")

        print(f"MAE  medio: {risultati['mae_mean']:.4f}  (std: {risultati['mae_std']:.4f}) "
              f"| miglior: {risultati['best_mae']:.4f}  | peggior: {risultati['worst_mae']:.4f}")

        print(f"MSE  medio: {risultati['mse_mean']:.4f}  (std: {risultati['mse_std']:.4f}) "
              f"| miglior: {risultati['best_mse']:.4f}  | peggior: {risultati['worst_mse']:.4f}")

        # R² potrebbe non essere presente nei risultati vecchi, gestiamo entrambi i casi
        if "r2_mean" in risultati:
            print(f"R²   medio: {risultati['r2_mean']:.4f}  (std: {risultati['r2_std']:.4f}) "
                  f"| miglior: {risultati['best_r2']:.4f}  | peggior: {risultati['worst_r2']:.4f}")


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

def ottieni_grafici_features_selection(n_features_range, mae_scores, mse_scores, r2_scores, nomeImg):
    os.makedirs("grafici", exist_ok=True)

    # --- Grafico MAE + R² sullo stesso plot ---
    fig, ax_mae = plt.subplots(figsize=(10, 5))

    # asse sinistro: MAE (blu)
    color_mae = 'steelblue'
    ax_mae.plot(n_features_range, mae_scores, marker='o', color=color_mae, label='MAE')
    ax_mae.set_xlabel("Numero di feature")
    ax_mae.set_ylabel("MAE", color=color_mae)
    ax_mae.tick_params(axis='y', labelcolor=color_mae)

    # evidenzia il minimo MAE
    idx_min_mae = int(np.argmin(mae_scores))
    ax_mae.axvline(x=n_features_range[idx_min_mae], color=color_mae,
                   linestyle='--', alpha=0.5, label=f'Miglior MAE ({n_features_range[idx_min_mae]} feat)')

    # asse destro: R² (verde)
    ax_r2 = ax_mae.twinx()
    color_r2 = 'seagreen'
    ax_r2.plot(n_features_range, r2_scores, marker='s', color=color_r2, label='R²')
    ax_r2.set_ylabel("R²", color=color_r2)
    ax_r2.tick_params(axis='y', labelcolor=color_r2)

    # evidenzia il massimo R²
    idx_max_r2 = int(np.argmax(r2_scores))
    ax_r2.axvline(x=n_features_range[idx_max_r2], color=color_r2,
                  linestyle='--', alpha=0.5, label=f'Miglior R² ({n_features_range[idx_max_r2]} feat)')

    # legenda unificata dai due assi
    linee_mae, etichette_mae = ax_mae.get_legend_handles_labels()
    linee_r2,  etichette_r2  = ax_r2.get_legend_handles_labels()
    ax_mae.legend(linee_mae + linee_r2, etichette_mae + etichette_r2, loc='lower right')

    ax_mae.set_title(f"MAE e R² vs Numero di feature\n{nomeImg}")
    ax_mae.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"grafici/{nomeImg}_mae_r2.png", dpi=300, bbox_inches="tight")
    print(f"Grafico MAE+R² salvato per '{nomeImg}'")
    plt.close(fig)

    # --- Grafico MSE separato (opzionale, lo teniamo per completezza) ---
    fig_mse, ax_mse = plt.subplots(figsize=(8, 4))
    ax_mse.plot(n_features_range, mse_scores, marker='o', color='tomato', label='MSE')
    idx_min_mse = int(np.argmin(mse_scores))
    ax_mse.axvline(x=n_features_range[idx_min_mse], color='tomato',
                   linestyle='--', alpha=0.5, label=f'Miglior MSE ({n_features_range[idx_min_mse]} feat)')
    ax_mse.set_xlabel("Numero di feature")
    ax_mse.set_ylabel("MSE")
    ax_mse.set_title(f"MSE vs Numero di feature - {nomeImg}")
    ax_mse.legend()
    ax_mse.grid(True, alpha=0.3)
    fig_mse.tight_layout()
    fig_mse.savefig(f"grafici/{nomeImg}_mse.png", dpi=300, bbox_inches="tight")
    plt.close(fig_mse)

    plt.close('all')