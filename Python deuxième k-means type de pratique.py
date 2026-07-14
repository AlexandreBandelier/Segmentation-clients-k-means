import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# --- 1. CONFIGURATION DES CHEMINS ---
dossier_actuel = os.path.dirname(os.path.abspath(__file__))

# Fichier 1 : Tes calculs RFM existants (contenant l'e-mail, la récence, les commandes, la dépense)
chemin_rfm = os.path.join(dossier_actuel, 'donnees_boutique_propres_interne - donnees_boutique_propres_interne.csv')

# Fichier 2 : Ton fichier de transactions/profils (contenant l'e-mail et les colonnes d'articles #1 à #10)
chemin_transactions = os.path.join(dossier_actuel, 'export-des-commandes-2018-13_07_2026 (sans .xlsx - Sans note du client.csv')

# Fichier de sortie final
chemin_sortie = os.path.join(dossier_actuel, 'segmentation_rfm_complete.csv')

print("Étape 1 : Chargement des deux fichiers de données...")

# Vérification de la présence des fichiers
for chemin, nom in [(chemin_rfm, "RFM"), (chemin_transactions, "Transactions")]:
    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"Le fichier '{chemin}' ({nom}) est introuvable.\n"
            "Assure-toi de le placer dans le même dossier que ce script."
        )

df_rfm = pd.read_csv(chemin_rfm)
df_trans = pd.read_csv(chemin_transactions)

# --- 2. HARMONISATION DES NOMS DE COLONNES COMMUNES (E-MAIL) ---
# Nous devons nous assurer que la colonne de jointure a exactement le même nom.
# On cherche une colonne contenant 'mail' ou 'client' dans chaque fichier pour la renommer en 'Email'
for df in [df_rfm, df_trans]:
    col_email = [col for col in df.columns if 'mail' in col.lower() or 'client' in col.lower()]
    if col_email:
        df.rename(columns={col_email[0]: 'Email'}, inplace=True)
    else:
        raise KeyError("Impossible de trouver une colonne d'identifiant (Email ou Client) dans l'un des fichiers.")

# --- 3. FUSION DES DEUX FICHIERS (JOINTURE SUR L'EMAIL) ---
print("Étape 2 : Fusion (Merge) des données sur la clé 'Email'...")
# On utilise un 'left join' pour garder tous nos clients RFM et y greffer les infos du catalogue si disponibles
df_fusion = pd.merge(df_rfm, df_trans, on='Email', how='left')

# --- 4. APPLICATION DE TA LOGIQUE DE SEGMENTATION EXISTANTE ---
print("Étape 3 : Application de ta logique de catégorisation actuelle...")

def attribuer_segment(row):
    # Gestion des variations de noms de colonnes possibles dans ton fichier RFM
    commandes = row.get('Commandes', row.get('Frequence', 0))
    depense = row.get('Dépense totale', row.get('Montant', 0))
    recence = row.get('Recence_Jours', row.get('Recence', 999))
    
    # 1. Traitement de la récence (plus de 6 mois d'inactivité)
    if recence > 180:
        if commandes >= 5 and depense >= 300:
            return 'Ancien VIP Dormant'
        return 'À relancer / Dormant'
    
    # 2. Traitement des profils actifs
    elif commandes >= 5 and depense >= 300:
        return 'VIP / Passionné'
    elif commandes == 1 and recence <= 30:
        return 'Nouveau client'
    else:
        return 'Client régulier'

# Création de ta colonne de segmentation métier
df_fusion['Segment_Metier'] = df_fusion.apply(attribuer_segment, axis=1)

# --- 5. PRÉPARATION ET CLUSTERING K-MEANS ---
print("Étape 4 : Préparation des données pour le K-Means...")

# Récupération dynamique des colonnes de ton RFM pour le modèle
col_recence = 'Recence_Jours' if 'Recence_Jours' in df_fusion.columns else 'Recence'
col_frequence = 'Commandes' if 'Commandes' in df_fusion.columns else 'Frequence'
col_montant = 'Dépense totale' if 'Dépense totale' in df_fusion.columns else 'Montant'

features_rfm = [col_recence, col_frequence, col_montant]

# Nettoyage des outliers (K-Means y est très sensible)
seuil_montant = df_fusion[col_montant].quantile(0.99)
seuil_frequence = df_fusion[col_frequence].quantile(0.99)

df_fusion['Montant_Clean'] = np.clip(df_fusion[col_montant], 0, seuil_montant)
df_fusion['Frequence_Clean'] = np.clip(df_fusion[col_frequence], 0, seuil_frequence)
df_fusion['Recence_Clean'] = df_fusion[col_recence]

features_clean = ['Recence_Clean', 'Frequence_Clean', 'Montant_Clean']

# Standardisation (Scaling)
scaler = StandardScaler()
donnees_standardisees = scaler.fit_transform(df_fusion[features_clean].fillna(0))

# Entraînement du K-Means (5 Groupes)
print("Étape 5 : Calcul du K-Means (K=5)...")
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, random_state=42)
df_fusion['KMeans_Cluster'] = kmeans.fit_predict(donnees_standardisees)

# --- 6. EXPORT ET SYNTHÈSE ---
# On sauvegarde le fichier final fusionné avec les deux segmentations côte à côte
df_fusion.to_csv(chemin_sortie, index=False, float_format="%.2f")
print(f"\n-> Succès ! Ton fichier complet est généré ici : {chemin_sortie}")

# Matrice de confrontation : pour voir comment tes segments se répartissent dans les clusters mathématiques
print("\n=== CONFRONTATION : TES SEGMENTS MÉTIERS vs LES CLUSTERS K-MEANS ===")
matrice = pd.crosstab(df_fusion['Segment_Metier'], df_fusion['KMeans_Cluster'])
print(matrice)
