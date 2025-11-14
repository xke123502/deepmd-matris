#!/bin/bash
# 回退 dp --pt test 预计算图支持的改动
# 日期: 2025-11-14

echo "开始回退 dp --pt test 预计算图支持的改动..."

# 文件路径
MODEL_E0="/aisi/mnt/data_nas/jwzhou/opt/deepmd-matris/deepmd_matris/model_e0.py"
DEEP_EVAL="/aisi/mnt/data_nas/jwzhou/deepmd-kit/deepmd/pt/infer/deep_eval.py"
TEST_PY="/aisi/mnt/data_nas/jwzhou/deepmd-kit/deepmd/entrypoints/test.py"

# 备份文件
echo "创建备份..."
cp "$MODEL_E0" "${MODEL_E0}.backup.$(date +%Y%m%d_%H%M%S)"
cp "$DEEP_EVAL" "${DEEP_EVAL}.backup.$(date +%Y%m%d_%H%M%S)"
cp "$TEST_PY" "${TEST_PY}.backup.$(date +%Y%m%d_%H%M%S)"

echo "备份完成！"
echo ""
echo "请手动回退以下改动："
echo ""
echo "1. $MODEL_E0:"
echo "   - 恢复 set_systems 方法签名（删除 test_systems 参数）"
echo "   - 恢复 _load_precomputed_graphs 方法签名（删除 is_test 参数）"
echo "   - 删除 forward 方法中的 is_test 参数"
echo ""
echo "2. $DEEP_EVAL:"
echo "   - 删除 __init__ 中的 _current_test_sid 属性"
echo "   - 恢复 _eval_model 中的直接调用（删除测试参数传递逻辑）"
echo ""
echo "3. $TEST_PY:"
echo "   - 删除设置 test_systems 的代码块（第 122-126 行）"
echo "   - 删除设置 _current_test_sid 的代码块（第 132-135 行）"
echo ""
echo "详细改动列表请参考: DP_TEST_PRECOMPUTED_GRAPH_CHANGES.md"

