pacman::p_load(dplyr,tidyr,ggplot2,hrbrthemes,data.table,tools,utils)

l <- list.files(path = "data/",pattern="*.csv",full.names = TRUE)

df_list <- lapply(l,data.table::fread)

l <- l %>% 
  as_tibble() %>% 
  mutate(value=file_path_sans_ext(value)) %>% 
  separate(value,sep="_",into=c(NA,"ID")) 

names(df_list) <- l$ID
df<-dplyr::bind_rows(df_list, .id = "id") 

l <- data.frame(value=l$ID,id=seq(1:16))
l$value <- as.numeric(l$value)
df$id <- as.numeric(df$id)

df <- df %>% 
  left_join(l,by = "id")

df$value <- NULL
vroom::vroom_write(df,file = "data/data_file.tsv.xz")
