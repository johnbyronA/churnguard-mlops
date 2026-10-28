# LinkedIn post drafts

## Español

Un modelo con buen AUC no le ahorra un peso a nadie si nadie sabe cuándo actuar, ni cuándo deja de funcionar. 📉

Por eso construí **ChurnGuard**: un sistema MLOps de punta a punta para predecir abandono de clientes, pensado como si fuera a producción mañana.

Lo que más me gustó del resultado: sobre clientes que el modelo nunca vio, una campaña de retención que contacta a todos **pierde $45.759**. Con el modelo y el umbral típico de 0.5 gana $29.415. Con un umbral elegido para **maximizar el valor de negocio** gana **$36.384, un 24% más**, contactando solo al 28% de los clientes.

Lo que hay debajo:
🔹 Contrato de datos que bloquea datos rotos antes de entrenar
🔹 Optuna + MLflow: cada experimento queda trazado y registrado
🔹 Probabilidades calibradas y umbral optimizado por dinero, no por 0.5
🔹 Registro campeón/retador con compuerta de calidad: un modelo nuevo solo sale a producción si le gana al actual en el mismo conjunto de prueba
🔹 API en FastAPI que explica cada predicción con SHAP y recomienda la siguiente acción
🔹 Monitoreo de drift (PSI, KS y drift de predicciones)
🔹 Reentrenamiento continuo con GitHub Actions que abre un PR con el nuevo modelo
🔹 Docker, Prometheus, Grafana, 35 pruebas y 94% de cobertura

Un detalle que me tomó tiempo y vale la pena: al reentrenar con datos nuevos, el campeón se estaba evaluando sobre clientes con los que ya había entrenado. La solución fue asignar cada cliente a train/test con un hash de su ID, para que la comparación siempre sea justa.

Pruébalo tú mismo, el modelo corre en tu navegador 👇
https://johnbyrona.github.io/churnguard-mlops/

Código: https://github.com/johnbyronA/churnguard-mlops

¿Cómo deciden en sus equipos cuándo reentrenar un modelo?

#MLOps #MachineLearning #DataScience #Python #MLflow #FastAPI #Analytics

---

## English

A model with a great AUC saves nobody a dollar if no one knows when to act, or when it stops working. 📉

So I built **ChurnGuard**: an end-to-end MLOps system for customer churn, designed as if it were shipping to production tomorrow.

The part I like most: on customers the model never saw, a retention campaign that contacts everyone **loses $45,759**. With the model at the usual 0.5 threshold it earns $29,415. With a threshold chosen to **maximise business value** it earns **$36,384, 24% more**, while contacting only 28% of customers.

Under the hood:
🔹 A data contract that blocks broken data before training
🔹 Optuna + MLflow: every experiment tracked and registered
🔹 Calibrated probabilities and a threshold optimised for money, not 0.5
🔹 Champion/challenger registry with a quality gate: a new model only ships if it beats the current one on the same holdout
🔹 A FastAPI service that explains every prediction with SHAP and recommends the next action
🔹 Drift monitoring (PSI, KS and prediction drift)
🔹 Continuous training in GitHub Actions that opens a PR with the new model
🔹 Docker, Prometheus, Grafana, 35 tests, 94% coverage

One detail that took time and was worth it: when retraining on new data, the champion was being scored on customers it had already trained on. The fix was assigning each customer to train/test by a hash of their ID, so the comparison is always fair.

Try it yourself, the model runs in your browser 👇
https://johnbyrona.github.io/churnguard-mlops/

Code: https://github.com/johnbyronA/churnguard-mlops

How does your team decide when to retrain a model?

#MLOps #MachineLearning #DataScience #Python #MLflow #FastAPI #Analytics

---

## Suggested visuals (carousel or images)

1. A 20–30 s screen recording of the live demo: change a customer, then drag the threshold (lead with this one)
1. `reports/figures/value_curve.png` — the money chart
2. The Mermaid architecture diagram (screenshot from the README on GitHub)
3. `reports/drift_report.html` — screenshot of the "RETRAIN REQUIRED" verdict
4. The GitHub Actions job summary with the quality-gate checklist
5. `reports/figures/feature_importance.png`
