# Caricare i pacchetti necessari
library(readr)
library(dplyr)
library(stringr)  
library(GOSemSim, lib.loc="~/R/library")  
library(org.Hs.eg.db, lib.loc="~/R/library")
library(Rcpp, lib.loc="~/R/library")
library(BiocParallel)  # Per il parallelismo

# Configurare il numero di core da usare per il parallelismo
num_cores <- parallel::detectCores() - 2  # Usa tutti i core meno uno
param <- MulticoreParam(workers = num_cores)  # Configura il parallelismo

# Leggere il file CSV
df <- read_csv("/home/developer/Documents/Tesi/files/networks/union_nodes.csv")

# Selezionare solo le colonne di interesse
df_selected <- df %>% dplyr::select(name, `Entry Name`, `Gene Ontology IDs`)

# Trasformare le GO in una lista di vettori separando i termini con ";"
df_selected <- df_selected %>% 
  mutate(GO_list = lapply(str_split(`Gene Ontology IDs`, ";"), str_trim))

# Rimuovere eventuali proteine senza annotazioni GO
df_selected <- df_selected %>% filter(lengths(GO_list) > 0)

# Inizializzare un oggetto GOSemSim per la similarità GO su organismi umani (9606)
hsGO_BP <- godata(annoDb = "org.Hs.eg.db", ont = "BP", computeIC = TRUE)
hsGO_MF <- godata(annoDb = "org.Hs.eg.db", ont = "MF", computeIC = TRUE)
hsGO_CC <- godata(annoDb = "org.Hs.eg.db", ont = "CC", computeIC = TRUE)

print("hsGO loaded")

# Funzione per calcolare la similarità tra due proteine usando BMA con Lin
calc_similarity <- function(idx) {
  i <- idx[1]
  j <- idx[2]
  
  go_list1 <- df_selected$GO_list[[i]]
  go_list2 <- df_selected$GO_list[[j]]
  
  if (length(go_list1) == 0 || length(go_list2) == 0) {
    return(list(i, j, NA))
  }
  
  sim_BP <- mgoSim(go_list1, go_list2, semData=hsGO_BP, measure="Lin", combine="BMA")
  sim_MF <- mgoSim(go_list1, go_list2, semData=hsGO_MF, measure="Lin", combine="BMA")
  sim_CC <- mgoSim(go_list1, go_list2, semData=hsGO_CC, measure="Lin", combine="BMA")
  
  return(list(i, j, mean(c(sim_BP, sim_MF, sim_CC), na.rm = TRUE)))
}


# Ottenere la lista dei nomi delle proteine
proteins <- df_selected$name
n <- length(proteins)

# Creare una matrice quadrata di similarità
similarity_matrix <- matrix(NA, nrow=n, ncol=n, dimnames=list(proteins, proteins))

# Generare gli indici per il triangolo superiore della matrice
index_pairs <- expand.grid(1:n, 1:n)
index_pairs <- index_pairs[index_pairs$Var1 <= index_pairs$Var2, ]  # Solo triangolo superiore

# Inizio calcolo della similarità
print("Inizio calcolo delle similarità...")
start_time <- Sys.time()  # Tempo di inizio
total_comparisons <- nrow(index_pairs)  # Numero totale di confronti

# Funzione per il logging dell'avanzamento
progress_log <- function(counter, total) {
  elapsed_time <- difftime(Sys.time(), start_time, units = "mins")
  estimated_total_time <- (elapsed_time / counter) * total
  remaining_time <- estimated_total_time - elapsed_time
  
  log_message <- (sprintf("Progresso: %d/%d (%.2f%%) | Tempo trascorso: %.2f min | Stima tempo rimanente: %.2f min",
                  counter, total, (counter / total) * 100,
                  elapsed_time, remaining_time))
  
  write(log_message, file = "/var/log/R/parallel_progress.log", append = TRUE)
}

# Parallelizzazione del calcolo della similarità
results <- bplapply(1:nrow(index_pairs), function(k) {
  if (k %% 100 == 0) progress_log(k, total_comparisons)  # Log ogni 100 confronti
  calc_similarity(c(index_pairs$Var1[k], index_pairs$Var2[k]))
}, BPPARAM = param)

# Inserire i risultati nella matrice
for (res in results) {
  i <- res[[1]]
  j <- res[[2]]
  similarity_matrix[i, j] <- res[[3]]
  similarity_matrix[j, i] <- res[[3]]  # Matrice simmetrica
}

# Stampare una parte della matrice per verifica
print(similarity_matrix[1:5, 1:5])

# Salvare la matrice in un file CSV
write.csv(similarity_matrix, "/home/developer/Documents/Tesi/files/networks/similarity_matrix_all_mean.csv", row.names=TRUE)

print("Calcolo completato!")
