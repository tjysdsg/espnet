pipeline() {
    echo '================== ONEHOT TREE =================='
    local/train_test_scoring_model.sh $1 || exit 1
    
    echo '================== PROB TREE =================='
    local/train_test_scoring_model.sh $1 --scoring_opts '--use-probs' || exit 1
    
    echo '================== ONEHOT PP TREE =================='
    local/train_test_scoring_model.sh $1 --scoring_opts '--per-phone-clf' || exit 1
    
    echo '================== PROB PP TREE =================='
    local/train_test_scoring_model.sh $1 --scoring_opts '--use-probs --per-phone-clf' || exit 1
    
    echo '================== ONEHOT MLP =================='
    local/train_test_scoring_model.sh $1 --scoring_opts '--use-mlp' || exit 1
    
    echo '================== PROB MLP =================='
    local/train_test_scoring_model.sh $1 --scoring_opts '--use-probs --use-mlp' || exit 1
}

# ========== so762 + aug ==========
data_opts='--train_sets "so762_train" --test_sets "so762_test"'

# ========== librispeech + aug ==========
# data_opts='--train_sets "libri_scoring_train" --test_sets "libri_scoring_test"'

pipeline $data_opts
