#include "common.h"
#include "llama.h"
#include "build-info.h"

#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) max_logit = std::max(max_logit, v);
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) probs[i] /= sum_exp;
    return probs;
}

void perplexity(llama_context* ctx, const gpt_params& params, const std::string& inputFilePath) {
    std::ifstream inputFile(inputFilePath);
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open input file: " << inputFilePath << std::endl;
        return;
    }

    std::string line;
    int promptCount = 0;

    double nll = 0.0;
    int count = 0;

    while (std::getline(inputFile, line)) {
        promptCount++;
        auto tokens = ::llama_tokenize(ctx, line, true);

        const int n_vocab = llama_n_vocab(ctx);
        const int n_batch = params.n_batch;

        const int n_ctx = tokens.size();
        const int num_batches = (n_ctx + n_batch - 1) / n_batch;

        std::vector<float> logits;

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = j * n_batch;
            const int batch_size = std::min(n_ctx - batch_start, n_batch);

            // save original token and restore it after eval
            const auto token_org = tokens[batch_start];

            // add BOS token for the first batch of each chunk
            if (j == 0) {
                tokens[batch_start] = llama_token_bos();
            }

            if (llama_eval(ctx, tokens.data() + batch_start, batch_size, j * n_batch, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return;
            }

            // restore the original token in case it was set to BOS
            tokens[batch_start] = token_org;

            const auto batch_logits = llama_get_logits(ctx);
            logits.insert(logits.end(), batch_logits, batch_logits + batch_size * n_vocab);
        }

        for (int j = std::min(512, n_ctx / 2); j < n_ctx - 1; ++j) {
            // Calculate probability of next token, given the previous ones.
            const std::vector<float> tok_logits(
                logits.begin() + (j + 0) * n_vocab,
                logits.begin() + (j + 1) * n_vocab);

            const float prob = softmax(tok_logits)[tokens[j + 1]];

            nll += -std::log(prob);
            ++count;
        }

        double perplexity = std::exp(nll / count);
        std::cout << "Prompt " << promptCount << " Perplexity: " << perplexity << std::endl;
    }

    inputFile.close();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./perplexity <model-params> <input-file>" << std::endl;
        return 1;
    }

    gpt_params params;

    params.n_batch = 512;
    if (gpt_params_parse(argc - 1, argv + 1, params) == false) {
        return 1;
    }

    params.perplexity = true;
    params.n_batch = std::min(params.n_batch, params.n_ctx);

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model might not support context sizes greater than 2048 tokens (%d specified);"
            "expect poor results\n", __func__, params.n_ctx);
    }

    fprintf(stderr, "%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_backend_init(params.numa);

    llama_model* model;
    llama_context* ctx;

    // load the model and apply lora adapter, if any
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
            params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    perplexity(ctx, params, argv[2]);

    llama_print_timings(ctx);
    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
