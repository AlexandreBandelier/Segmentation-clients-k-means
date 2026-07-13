# Segmentation-clients-k-means/NLP/ACP
Pipeline ETL en Python pour la segmentation de clients e-commerce (K-means, NLP & ACP)

## Vision du Projet : "L'entreprise dans l'entreprise"

En ecommerce, l'envoi de messages génériques et uniformes détruit l'engagement client et réduit les marges. Ce projet vise à transformer une base de données transactionnelle WooCommerce brute en segments-clients quasi autonomes (i.e. habitudes de consommation presque exclusives, rendant possible un business plan d'entreprise complet sur un segment donné (pratiquants débutants, pratiquants de combat, pratiquants kata/traditionnel, etc.). Chaque segment identifié est donc doté de sa propre logique d'achat et de ses caractéristiques comportementales.

Pour atteindre cet objectif, ce projet a suivi une méthodologie d'apprentissage et d'optimisation en deux grandes étapes successives.

## Étape 1 : Le clustering K-Means financier classique (K=5)

1. Objectif & données d'entrée

L'objectif de cette première étape était de reprendre tes précédents résultats RFM (Récence, Fréquence, Montant) et d'y appliquer l'algorithme K-Means pour regrouper automatiquement tes clients selon leurs chiffres d'achat.

Le pipeline de données a fusionné :

Le fichier de calculs RFM clients : donnees_boutique_propres_interne.csv

Le détail des commandes brutes WooCommerce : export-des-commandes-2018-13_07_2026 (sans .xlsx - Sans note du client.csv

2. Pipeline Technique

- Normalisation financière : Nettoyage des chaînes textuelles complexes exportées par WooCommerce (retrait des symboles €, des virgules , décimales, et des espaces insécables de milliers \xa0).
- Traitement des Outliers : Écrêtage (clipping) des valeurs extrêmes de montant et de fréquence au 99ème percentile pour stabiliser l'algorithme K-Means.
- Mise à l'échelle (StandardScaler) : Standardisation des critères pour équilibrer le poids de la récence (jours) et du montant (euros).
- Clustering : Application d'un algorithme K-Means classique à 5 clusters.
- Limites identifiées : Ce premier modèle isole très bien la valeur financière mais est incapable de comprendre ce que les clients achètent réellement (disciplines de combat, profils d'âge, types d'équipements).

## Étape 2 : Le Clustering Comportemental Profond par ACP & NLP (K=6)

Pour surmonter les limites de l'Étape 1 et en l'absence de colonnes de catégories ou de tailles structurées, nous avons développé une approche de Text Mining sémantique connectée à un catalogue de produits.

1. Liaison intelligente avec le catalogue produits
Le fichier des commandes WooCommerce ne contient pas directement les catégories de sport (Kumite, Kata, etc.). Pour résoudre ce problème, le programme va chercher ces informations directement dans le catalogue de l'entreprise (export_produits_KGI.csv).

Le script associe automatiquement chaque article commandé au bon produit du catalogue, même si le titre de la commande contient des détails en plus (comme des tailles). Il compare les titres du plus long au plus court (i.e. ne prenant en compte que la partie commune du titre d'un fichier à l'autre pour retirer le suffixe éventuel de la taille ou d'autres spécificités) pour s'assurer de faire la liaison la plus précise possible et éviter les erreurs de correspondance.

2. Recherche Linguistique (Text Mining / NLP) sur toute la ligne

Pour chaque produit identifié, le script analyse l'intégralité du texte descriptif du catalogue (titres, tags, descriptions) à la recherche de mots-clés spécifiques :

- Kumite (Combat) : gant, plastron, protection, protege, combat, etc.
- Kata (Technique) : kata, lourd, claquant, traditionnel, gi-lourd, etc.
- Débutant (Initiation) : initiation, debutant, blanche, jaune, etc.
- Enfant (Junior) : enfant, kids, junior, fillette, garconnet, ado, etc.

Le modèle calcule ensuite des taux d'affinité (%) (ex : si un client a acheté 4 protections sur 5 articles au total, il possède 80% d'affinité Kumite (i.e. combat)).

3. Éviter que les gros budgets ne cachent les habitudes sportives (l'effet ACP)

En combinant le RFM (3 critères) et les affinités produits (4 critères), nous obtenons 7 dimensions. Pour éviter que les critères financiers ne soient dilués par les variables d'affinités (car les critères financiers sont composés de populations de plus petite taille), nous appliquons une Analyse en Composantes Principales (ACP/PCA) à 3 dimensions pour projeter nos données de façon géométriquement équilibrée avant de lancer le K-Means Profond (K=6) (i.e. on obtient certes des segments par affinité, mais aussi par budgets).

## Cartographie des 6 clusters

L'algorithme final à 6 clusters a révélé la structure logique suivante :

### Cluster #0 : Le pratiquant occasionnel éloigné (7 013 clients)

Caractéristiques : Dépense faible (138,09 €), récence très dégradée (1409 jours d'inactivité).

Style d'achat : Achat de commodité unique ou très irrégulier pour tester le sport il y a plusieurs années avant de s'en détacher.

### Cluster #1 : Le profil famille & pratique associative (2 032 clients)

Caractéristiques : Dépense modérée, récence récente (591 jours), 9,1% d'affinité "Enfant".

Style d'achat : Parents d'élèves ou pratiquants de clubs polyvalents qui s'équipent en famille au moment de la rentrée scolaire.

### Cluster #2 : L'anseignant, le club & le pratiquant élite (643 clients)

Caractéristiques : Panier d'achat très élevé de 1 014,91 € en moyenne pour 7,2 commandes/

Style d'achat : Dirigeants de dojos ou compétiteurs élites exigeant l'excellence technique (kimonos claquants de Kata, protections homologuées) et achetant en gros volumes.

### Cluster #3 : L'initié en phase de découverte (829 clients)

Caractéristiques : Dépense de 110,04 €, 21,2% d'affinité "Débutant" (ceintures de couleurs de départ, packs d'initiation).

Style d'achat : Pratiquant en phase de découverte qui s'équipe du kit de base exigé par son club sans investissement lourd au départ.

### Cluster #4 : Le petit acheteur (9 487 clients)

Caractéristiques : Panier moyen de 9,54 € pour 0,3 commande. Récence de 1019 jours.

Style d'achat : Achat ponctuel pour remplacer un petit accessoire perdu (gourde, protège-dents d'urgence, écusson).

### Cluster #5 : Le compétiteur kumite (i.e. combat) (2 095 clients)

Caractéristiques : Dépense de 396,67 € sur 3,2 commandes, 19,7% d'affinité Kumite.

Style d'achat : Pratiquant régulier focalisé sur le combat, renouvelant fréquemment son matériel de protection homologué WKF en raison de l'usure des entraînements.

## Visualiseur de segmentation interactif (visualiseur_interactive.html)

Vous pouvez accéder à l'application web interactive permettant d'accéder au tableau de bord visuel des clusters, avec le fichier visualiseur_clusters_interactif.html :
![Aperçu du Tableau de Bord](images/visualisation_clusters.png)
