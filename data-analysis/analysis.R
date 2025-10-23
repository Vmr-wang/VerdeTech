# analysis.R — per-task (LR+DT 合并为 Classification)，含相关性与效应量

suppressPackageStartupMessages({
  library(tidyverse)
  library(readr)
  library(janitor)
  library(rstatix)
  library(effsize)
  library(patchwork)
  suppressWarnings(suppressMessages(requireNamespace("FSA", quietly = TRUE)))
})

dir.create("data",    showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)
dir.create("figs",    showWarnings = FALSE)

# ---------- Read ----------
df_raw <- readr::read_csv("data/run_table.csv",
                          show_col_types = FALSE,
                          na = c("", "NA")) %>%
  janitor::clean_names()

# ---------- Types ----------
to_num <- c("actual_size","n_features","cpu_energy","cpu_util","gpu_energy",
            "gpu_util","gpu_power_avg","runtime","memory","accuracy","mse",
            "cpu_temp_before","gpu_temp_before","cpu_temp_after","gpu_temp_after")
for (v in intersect(to_num, names(df_raw))) {
  df_raw[[v]] <- suppressWarnings(as.numeric(df_raw[[v]]))
}

# ---------- Derive ----------
df <- df_raw %>%
  mutate(
    alg = factor(alg),
    impl = factor(impl),
    dataset = factor(dataset),
    task = case_when(
      as.character(alg) %in% c("LR","DT") ~ "Classification",
      as.character(alg) %in% c("RR")      ~ "Regression",
      as.character(alg) %in% c("KMeans")  ~ "Clustering",
      TRUE ~ as.character(alg)
    ),
    task = factor(task, levels = c("Classification","Regression","Clustering")),
    energy_total = coalesce(cpu_energy, 0) + coalesce(gpu_energy, 0),
    power_avg_total    = if_else(runtime > 0, energy_total / runtime, NA_real_),
    power_avg_cpu      = if_else(runtime > 0, cpu_energy  / runtime, NA_real_),
    power_avg_gpu_calc = if_else(runtime > 0, gpu_energy  / runtime, NA_real_),
    power_avg_gpu_meas = gpu_power_avg,
    mem_gib  = memory / (1024^3),
    cpu_share = if_else(energy_total > 0, cpu_energy / energy_total, NA_real_),
    gpu_share = if_else(energy_total > 0, gpu_energy / energy_total, NA_real_)
  ) %>%
  filter(
    is.na(runtime) | runtime >= 0,
    is.na(cpu_energy) | cpu_energy >= 0,
    is.na(gpu_energy) | gpu_energy >= 0
  )

# ---------- Metrics ----------
metrics_common <- intersect(c(
  "energy_total","cpu_energy","gpu_energy",
  "power_avg_total","power_avg_cpu","power_avg_gpu_calc","power_avg_gpu_meas",
  "runtime","mem_gib","cpu_util","gpu_util"
), names(df))
score_candidates <- intersect(c("accuracy","mse"), names(df))
metrics_all <- c(metrics_common, score_candidates)

# ---------- Outliers (global) ----------
mark_outlier <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  ok <- is.finite(x)
  out <- rep(FALSE, length(x))
  if (sum(ok) < 4) return(out)
  qs <- stats::quantile(x[ok], probs = c(0.25, 0.75), na.rm = TRUE)
  iqr <- qs[[2]] - qs[[1]]
  lo <- qs[[1]] - 1.5 * iqr
  hi <- qs[[2]] + 1.5 * iqr
  out[ok] <- x[ok] < lo | x[ok] > hi
  out
}

df <- df %>%
  mutate(across(all_of(metrics_all), function(x) ifelse(is.finite(x), x, NA_real_))) %>%
  mutate(across(all_of(metrics_all), function(x) mark_outlier(x), .names = "is_out_{col}"))

run_id_col <- if ("__run_id" %in% names(df)) "__run_id" else if ("run_id" %in% names(df)) "run_id" else names(df)[1]
outlier_cols <- grep("^is_out_", names(df), value = TRUE)
df %>%
  transmute(run_id = .data[[run_id_col]], alg, task, impl, dataset, across(all_of(outlier_cols))) %>%
  write_csv("results/outliers_report.csv")

# ---------- Helpers ----------
safe_name <- function(x) gsub("[^A-Za-z0-9._-]", "_", as.character(x))
safe_min  <- function(x) { x <- x[is.finite(x)]; if (length(x)) min(x) else NA_real_ }
safe_max  <- function(x) { x <- x[is.finite(x)]; if (length(x)) max(x) else NA_real_ }

plot_metric_impl_by_dataset <- function(d, metric) {
  d2 <- d %>% filter(is.finite(.data[[metric]]))
  p_v <- ggplot(d2, aes(x = impl, y = .data[[metric]], fill = impl)) +
    geom_violin(trim = FALSE) +
    geom_boxplot(width = 0.15) +
    facet_wrap(~ dataset, scales = "free_y") +
    labs(x = "impl", y = metric, title = paste("Impl comparison by dataset —", metric)) +
    theme_bw() + theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none")
  p_d <- ggplot(d2, aes(x = .data[[metric]], fill = impl)) +
    geom_density(alpha = 0.4) +
    facet_wrap(~ dataset, scales = "free") +
    labs(x = metric, y = "Density", title = paste("Density by impl within dataset —", metric)) +
    theme_bw() + theme(legend.position = "bottom")
  list(v = p_v, d = p_d)
}

choose_score_metric <- function(d, candidates = c("accuracy","mse")) {
  available <- candidates[candidates %in% names(d)]
  if (length(available) == 0) return(character())
  has_nonzero <- sapply(available, function(m) {
    vals <- d[[m]][is.finite(d[[m]])]
    length(vals) > 0 && any(abs(vals) > 0, na.rm = TRUE)
  })
  if (any(has_nonzero)) return(available[which(has_nonzero)[1]])
  has_finite <- sapply(available, function(m) any(is.finite(d[[m]])))
  if (any(has_finite)) return(available[which(has_finite)[1]])
  character()
}

score_metric_for_task <- function(task, d) {
  t_chr <- as.character(task)[1]
  if (t_chr %in% c("Classification")) {
    if ("accuracy" %in% names(d)) return("accuracy")
    return(choose_score_metric(d, candidates = c("mse")))
  }
  if (t_chr %in% c("Regression")) {
    if ("mse" %in% names(d)) return("mse")
    return(choose_score_metric(d, candidates = c("accuracy")))
  }
  if (t_chr %in% c("Clustering")) {
    if ("accuracy" %in% names(d)) return("accuracy")
    return(choose_score_metric(d))
  }
  choose_score_metric(d)
}

eta2_safe <- function(aov_fit) {
  es <- tryCatch(rstatix::eta_squared(aov_fit, partial = FALSE), error = function(e) NULL)
  if (is.null(es)) return(NA_real_)
  if (is.atomic(es)) return(as.numeric(es)[1])
  if (is.data.frame(es) && "Eta2" %in% names(es)) return(es$Eta2[1])
  if (is.data.frame(es) && "eta.sq" %in% names(es)) return(es$eta.sq[1])
  suppressWarnings(as.numeric(es)[1])
}

# 成对效应量：Cliff's delta 与 Hedges' g
pair_es <- function(di, metric) {
  grps <- levels(factor(di$impl))
  out <- list()
  if (length(grps) < 2) return(tibble(contrast=character(), cliffs_delta=double(), hedges_g=double()))
  for (i in 1:(length(grps)-1)) for (j in (i+1):length(grps)) {
    g1 <- grps[i]; g2 <- grps[j]
    x <- di %>% filter(impl %in% c(g1,g2)) %>% select(impl, val = !!rlang::sym(metric))
    x <- x %>% filter(is.finite(val))
    if (n_distinct(x$impl) < 2) next
    cd <- tryCatch(effsize::cliff.delta(val ~ impl, data = x)$estimate, error=function(e) NA_real_)
    hg <- tryCatch(effsize::cohen.d(val ~ impl, data = x, hedges.correction = TRUE)$estimate, error=function(e) NA_real_)
    out <- append(out, list(tibble(contrast = paste(g1, g2, sep="-"),
                                   cliffs_delta = as.numeric(cd),
                                   hedges_g = as.numeric(hg))))
  }
  if (length(out)) bind_rows(out) else tibble(contrast=character(), cliffs_delta=double(), hedges_g=double())
}

# ---------- Tests: per dataset compare impl (returns decision log) ----------
stratified_test_impl <- function(d, metric) {
  dd <- d %>% filter(!is.na(.data[[metric]])) %>% drop_na(dataset, impl)
  out <- list()
  dec <- list()
  for (ds in unique(dd$dataset)) {
    di <- dd %>% filter(dataset == ds)
    if (n_distinct(di$impl) < 2) next

    ng <- di %>% group_by(impl) %>%
      summarise(
        p = tryCatch(rstatix::shapiro_test(.data[[metric]]) %>% dplyr::pull(p),
                     error = function(e) NA_real_),
        .groups="drop"
      ) %>% dplyr::pull(p)
    valid_p <- ng[is.finite(ng)]
    use_anova <- length(valid_p) >= 2 && mean(valid_p > 0.05) >= 0.5

    lev <- tryCatch(rstatix::levene_test(di, as.formula(paste(metric, "~ impl"))), error=function(e) NULL)
    lev_p <- if (!is.null(lev)) lev$p[1] else NA_real_

    if (use_anova) {
      fit <- aov(as.formula(paste(metric, "~ impl")), data = di)
      eta2 <- eta2_safe(fit)
      pval <- summary(fit)[[1]][["Pr(>F)"]][1]
      base <- tibble(dataset = ds, metric = metric, test = "ANOVA",
                     p = pval, effect = eta2, effect_type = "eta^2", parametric = TRUE)
      pw <- tryCatch({
        TukeyHSD(fit)[["impl"]] %>% as.data.frame() %>%
          rownames_to_column("contrast") %>% as_tibble() %>%
          transmute(dataset = ds, metric = metric, contrast, p = `p adj`, test = "TukeyHSD") %>%
          mutate(p_adj_BH = p.adjust(p, method = "BH"))
      }, error = function(e) tibble())
    } else {
      kw <- rstatix::kruskal_test(di, as.formula(paste(metric, "~ impl")))
      eps2 <- tryCatch({
        ef <- rstatix::kruskal_effsize(di, as.formula(paste(metric, "~ impl")))
        dplyr::coalesce(ef$effsize[1], ef$estimate[1])
      }, error = function(e) NA_real_)
      base <- tibble(dataset = ds, metric = metric, test = "Kruskal-Wallis",
                     p = kw$p, effect = eps2, effect_type = "epsilon^2", parametric = FALSE)
      pw <- tryCatch({
        if (requireNamespace("FSA", quietly = TRUE)) {
          dt <- FSA::dunnTest(as.formula(paste(metric, "~ impl")), data = di, method = "bh")
          as_tibble(dt$res) %>%
            transmute(dataset = ds, metric = metric,
                      contrast = Comparison, p = P.unadj, p_adj_BH = P.adj,
                      test = "Dunn (FSA, BH)")
        } else {
          pwil <- pairwise.wilcox.test(di[[metric]], di$impl, p.adjust.method = "BH")
          as.data.frame(as.table(pwil$p.value)) %>%
            filter(!is.na(Freq)) %>%
            transmute(dataset = ds, metric = metric,
                      contrast = paste(Var1, Var2, sep = "-"),
                      p = NA_real_, p_adj_BH = as.numeric(Freq),
                      test = "Pairwise Wilcoxon (BH)")
        }
      }, error = function(e) tibble())
    }

    # pairwise effect sizes
    es_tbl <- pair_es(di, metric)
    if (nrow(pw) > 0 && nrow(es_tbl) > 0) {
      pw <- pw %>% dplyr::left_join(es_tbl, by = "contrast")
    } else if (nrow(pw) > 0) {
      pw <- pw %>% mutate(cliffs_delta = NA_real_, hedges_g = NA_real_)
    }

    dec <- append(dec, list(tibble(
      dataset = ds, metric = metric,
      test_chosen = if (use_anova) "ANOVA" else "Kruskal",
      normality_ok_ratio = if (length(valid_p)) mean(valid_p > 0.05) else NA_real_,
      levene_p = lev_p,
      n_groups = dplyr::n_distinct(di$impl),
      n_total = nrow(di)
    )))
    out <- append(out, list(list(base = base, pw = pw)))
  }

  if (length(out) == 0) {
    return(list(main = tibble(), pairwise = tibble(), decision = tibble()))
  }
  list(
    main = bind_rows(lapply(out, function(z) z$base)),
    pairwise = bind_rows(lapply(out, function(z) z$pw)),
    decision = bind_rows(dec)
  )
}

# ---------- Per-task outputs ----------
groups <- unique(df$task)
for (g in groups) {
  g_str <- safe_name(g)
  dir_res <- file.path("results", g_str)
  dir_fig <- file.path("figs",    g_str)
  dir.create(dir_res, showWarnings = FALSE, recursive = TRUE)
  dir.create(dir_fig, showWarnings = FALSE, recursive = TRUE)

  df_g <- df %>% filter(task == g)

  # score metric per task
  score_metric <- score_metric_for_task(g, df_g)
  metrics_g <- metrics_common[metrics_common %in% names(df_g)]
  if (length(score_metric)) metrics_g <- unique(c(metrics_g, score_metric))

  # 0) Shapiro for score metric
  if (length(score_metric)) {
    norm_tbl <- df_g %>%
      select(dataset, impl, value = !!rlang::sym(score_metric)) %>%
      group_by(dataset, impl) %>%
      summarise(
        n = sum(is.finite(value)),
        p_shapiro = tryCatch(rstatix::shapiro_test(value) %>% dplyr::pull(p), error=function(e) NA_real_),
        .groups = "drop"
      )
    if (nrow(norm_tbl) > 0) {
      write_csv(norm_tbl, file.path(dir_res, "assumption_normality_by_dataset_impl.csv"))
    } else {
      write_lines("no normality results", file.path(dir_res, "assumption_normality_by_dataset_impl.EMPTY.txt"))
    }
  } else {
    write_lines("no score metric available", file.path(dir_res, "assumption_normality_by_dataset_impl.EMPTY.txt"))
  }

  # 1) Summary by dataset × impl
  summary_tbl_g <- df_g %>%
    summarise(
      .by = c(dataset, impl),
      n = n(),
      across(all_of(metrics_g), list(
        mean = ~mean(.x, na.rm = TRUE),
        sd   = ~sd(.x, na.rm = TRUE),
        var  = ~var(.x, na.rm = TRUE),
        median = ~median(.x, na.rm = TRUE),
        min = ~safe_min(.x),
        max = ~safe_max(.x)
      ), .names = "{.col}_{.fn}")
    )
  write_csv(summary_tbl_g, file.path(dir_res, "summary_stats_by_dataset_impl.csv"))

  # 2) Plots
  for (m in metrics_g) {
    gplt <- plot_metric_impl_by_dataset(df_g, m)
    ggsave(file.path(dir_fig, paste0(m, "_violin_impl_by_dataset.png")), gplt$v, width = 10, height = 6, dpi = 150)
    ggsave(file.path(dir_fig, paste0(m, "_density_impl_by_dataset.png")), gplt$d, width = 10, height = 6, dpi = 150)
  }

  # 3) Tests
  all_main <- list(); all_pw <- list(); all_dec <- list()
  for (m in metrics_g) {
    res <- stratified_test_impl(df_g, m)
    all_main[[m]] <- res$main
    all_pw[[m]]   <- res$pairwise
    all_dec[[m]]  <- res$decision
  }
  main_tbl <- bind_rows(all_main)
  pw_tbl   <- bind_rows(all_pw)
  dec_tbl  <- bind_rows(all_dec)

  if (!is.null(main_tbl) && ncol(main_tbl) > 0 && nrow(main_tbl) > 0) {
    main_tbl <- main_tbl %>%
      group_by(dataset, metric) %>%
      mutate(p_adj_BH = p.adjust(p, method = "BH")) %>%
      ungroup()
    write_csv(main_tbl, file.path(dir_res, "tests_impl_within_dataset_main.csv"))
  } else {
    write_lines("no tests produced", file.path(dir_res, "tests_impl_within_dataset_main.EMPTY.txt"))
  }

  if (!is.null(pw_tbl) && ncol(pw_tbl) > 0) {
    write_csv(pw_tbl, file.path(dir_res, "tests_impl_within_dataset_pairwise.csv"))
  } else {
    write_lines("no pairwise tests produced", file.path(dir_res, "tests_impl_within_dataset_pairwise.EMPTY.txt"))
  }

  if (!is.null(dec_tbl) && ncol(dec_tbl) > 0) {
    write_csv(dec_tbl, file.path(dir_res, "test_decision_by_dataset_metric.csv"))
  } else {
    write_lines("no decision log produced", file.path(dir_res, "test_decision_by_dataset_metric.EMPTY.txt"))
  }

  # 3b) Correlations for RQ2: energy_total vs runtime/mem/util/accuracy/mse
  corr_targets <- intersect(c("runtime","mem_gib","cpu_util","gpu_util","accuracy","mse"), names(df_g))
  if (length(corr_targets)) {
    cor_overall <- purrr::map_dfr(corr_targets, function(tg) {
      x <- df_g$energy_total; y <- df_g[[tg]]
      ok <- is.finite(x) & is.finite(y)
      if (sum(ok) < 3) return(tibble(metric=tg, scope="overall", rho=NA_real_, p=NA_real_, n=sum(ok)))
      cs <- suppressWarnings(cor.test(x[ok], y[ok], method="spearman", exact=FALSE))
      tibble(metric=tg, scope="overall", rho=unname(cs$estimate), p=cs$p.value, n=sum(ok))
    })

    cor_by_impl <- df_g %>%
      filter(is.finite(energy_total)) %>%
      select(impl, energy_total, all_of(corr_targets)) %>%
      pivot_longer(cols = all_of(corr_targets), names_to="metric", values_to="y") %>%
      group_by(impl, metric) %>%
      summarise(
        n = sum(is.finite(energy_total) & is.finite(y)),
        rho = if (n >= 3) suppressWarnings(cor(energy_total, y, method="spearman", use="complete.obs")) else NA_real_,
        p = if (n >= 3) suppressWarnings(cor.test(energy_total, y, method="spearman", exact=FALSE)$p.value) else NA_real_,
        .groups="drop"
      ) %>%
      mutate(scope = paste0("impl:", impl)) %>%
      select(metric, scope, rho, p, n)

    cor_tbl <- bind_rows(cor_overall, cor_by_impl) %>%
      group_by(metric) %>%
      mutate(p_adj_BH = p.adjust(p, method="BH")) %>%
      ungroup()

    readr::write_csv(cor_tbl, file.path(dir_res, "correlations_energy_vs_metrics.csv"))
  }

  # 4) Temperature deltas
  temp_tbl_g <- df_g %>%
    summarise(.by = c(dataset, impl),
              cpu_temp_delta = mean(cpu_temp_after - cpu_temp_before, na.rm = TRUE),
              gpu_temp_delta = mean(gpu_temp_after - gpu_temp_before, na.rm = TRUE))
  write_csv(temp_tbl_g, file.path(dir_res, "temp_deltas_by_dataset_impl.csv"))
}

# ---------- Session info ----------
sessionInfo() %>% capture.output() %>% writeLines(con = "results/sessionInfo.txt")
writeLines("Per-task analysis completed.", "results/DONE.txt")
