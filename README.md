## Introduzione
Il progetto è stato fatto per l'esame di ICON (ingegneria della conoscenza, nella triennale di informatica a UNIBA).
L'obiettivo è quello di dimostrare che sia possibile predire la "pericolosità" di un tratto di strada (sotto forma di incidenti annuali), considerando le caratteristiche strutturali del percorso e ciò che lo circonda tramite utilizzo di machine learning e reti Bayessiane.
Si è fatta sia regressione che classificazione la prima solo con modelli di ML tradizionale: Random Forest, SVR, K-NN, Gradient Boosting, ElasticNet, BayesianRidge;
invece la classificazione è fatta sia con modelli di ML: Random Forest, Gradient Boosting, SVC, KNN ma anche con reti bayessiane (necessarie per lo scopo di esplorare approcci diversi al problema)

I risultati ottenuti non sono particolarmente positivi, quindi il punto predisposto non è stato provato in queste condizioni (ma il principale focus dell'esame era l'utilizzo di algoritmi di apprendimento)

## Dataset
Il dataset viene sviluppato a partire dai dati degli incidenti accaduti su strade statali Italiane nel 2023, pubblicato da ACI (Autimobile Club Italia), il documento suddivide le strade sul chilometraggio, la regione e altre informazioni riguardo gli incidenti su quel tratto di strada. 
Per ottenere informazioni esterne a quel dataset in modo che siano reperibili per le predizioni si è deciso di usare l’ontologia stradale OpenStreetMap, accessibile tramite la libreria osmnx che permette di fare query ottenendo le informazioni sulla rete stradale, le sue caratteristiche e strutture presenti nel territorio.



## Modalità di utilizzo
Per utilizzare il software assicurarsi di avere installato le librerie in requirements.txt se non sono state installate andare da linea di comando,
spostarsi nella cartella contenente requirements.txt ed eseguire “pip install -r requirements.txt”.

Installati i ‘requirements’ o si procede aprendo il progetto con ide, andando al file main.py e avviare il programma,
oppure portarsi nella directory di main.py tramite cmd e procedere con il comando “python main.py”.
Le interazioni sono eseguite tramite inut o nella console o direttamente nella cmd a seconda della scelta fatta precedentemente.

## Per ulteriori informazioni vedere il file "documentazioneICON.docx" 
