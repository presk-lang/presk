/* llamacpp_logits.c — the logits llama.cpp computes on given ids (plan 0.17.1,
 * presk D185): the field's quantizations measured on the same tokens as
 * presk's realizations, no tokenizer between the tools.
 *
 *     llamacpp_logits MODEL.gguf IDS.txt OUT.f32 [THREADS]
 *
 * IDS.txt holds one window a line, its ids separated by spaces (the
 * beginning-of-text token first, as presk's samples). Each window is decoded
 * alone, from an empty cache, in one batch whose every position asks for its
 * logits; OUT.f32 receives them as float32, window after window, position
 * after position, the vocabulary's entries in order. The context's settings
 * are llama.cpp's defaults (its KV cache in f16, its flash attention `auto`)
 * but for the context's length, the batch's and the threads: what a user of
 * the library gets. A line on stderr gives the vocabulary, the windows and
 * the positions written.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"

static void quiet(enum ggml_log_level level, const char *text, void *data) {
    (void)data;
    if (level >= GGML_LOG_LEVEL_WARN)
        fputs(text, stderr);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.gguf IDS.txt OUT.f32 [THREADS]\n", argv[0]);
        return 2;
    }
    int threads = argc > 4 ? atoi(argv[4]) : 1;
    FILE *ids = fopen(argv[2], "r");
    if (!ids) {
        perror(argv[2]);
        return 1;
    }
    FILE *out = fopen(argv[3], "wb");
    if (!out) {
        perror(argv[3]);
        return 1;
    }
    llama_log_set(quiet, NULL);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "%s: cannot load\n", argv[1]);
        return 1;
    }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int nv = llama_vocab_n_tokens(vocab);

    enum { MAXT = 4096 };
    static char line[1 << 16];
    llama_token toks[MAXT];
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "cannot make a context\n");
        return 1;
    }
    struct llama_batch batch = llama_batch_init(512, 0, 1);
    long windows = 0, positions = 0;
    while (fgets(line, sizeof line, ids)) {
        int n = 0;
        for (char *p = strtok(line, " \t\n"); p; p = strtok(NULL, " \t\n")) {
            if (n == MAXT || n == 512) {
                fprintf(stderr, "window %ld: longer than the context\n", windows);
                return 1;
            }
            toks[n++] = (llama_token)atol(p);
        }
        if (n == 0)
            continue;
        llama_memory_clear(llama_get_memory(ctx), true);
        batch.n_tokens = n;
        for (int i = 0; i < n; i++) {
            batch.token[i] = toks[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "window %ld: llama_decode failed\n", windows);
            return 1;
        }
        for (int i = 0; i < n; i++) {
            const float *z = llama_get_logits_ith(ctx, i);
            if (!z || fwrite(z, sizeof(float), (size_t)nv, out) != (size_t)nv) {
                fprintf(stderr, "window %ld, position %d: no logits\n", windows, i);
                return 1;
            }
        }
        windows++;
        positions += n;
    }
    fclose(out);
    fprintf(stderr, "vocabulary %d, %ld windows, %ld positions, %d thread(s)\n", nv, windows, positions, threads);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
