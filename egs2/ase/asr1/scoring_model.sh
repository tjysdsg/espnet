pipeline() {
    data_opts="--train_sets \"$1\" --test_sets \"$2\""
    echo '================== ONEHOT TREE =================='
    local/train_test_scoring_model.sh ${data_opts} || exit 1
    
    echo '================== PROB TREE =================='
    local/train_test_scoring_model.sh ${data_opts} --scoring_opts '--use-probs' || exit 1
    
    echo '================== ONEHOT PP TREE =================='
    local/train_test_scoring_model.sh ${data_opts} --scoring_opts '--per-phone-clf' || exit 1
    
    echo '================== PROB PP TREE =================='
    local/train_test_scoring_model.sh ${data_opts} --scoring_opts '--use-probs --per-phone-clf' || exit 1
    
    echo '================== ONEHOT MLP =================='
    local/train_test_scoring_model.sh ${data_opts} --scoring_opts '--use-mlp' || exit 1
    
    echo '================== PROB MLP =================='
    local/train_test_scoring_model.sh ${data_opts} --scoring_opts '--use-probs --use-mlp' || exit 1
}

# ========== so762 + aug ==========
pipeline "so762_train" "so762_test"

# ========== librispeech + aug ==========
# pipeline "libri_scoring_train" "libri_scoring_test"
