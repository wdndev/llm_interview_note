# 企业C

# 一面

### 技术问题

1.  ChatGLM3 + RAG项目流程
2.  微调数据集如果获取，数据集如何构造为微调数据集？
3.  LoRA微调原理
4.  LoRA微调的主要参数有那几个？
    1.  主要三个：`r`、 $\alpha$、微调位置
5.  还了解其他微调的方式吗？

### Leetcode

[105. 从前序与中序遍历序列构造二叉树 - 力扣（LeetCode）](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/ "105. 从前序与中序遍历序列构造二叉树 - 力扣（LeetCode）")

```c++
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        return this->pre_inorder_build_tree(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
    }

private:
    TreeNode* pre_inorder_build_tree(std::vector<int>& preorder, int pre_start_idx, int pre_end_idx,
                                    std::vector<int>& inorder, int in_start_idx, int in_end_idx) {
        if (pre_start_idx > pre_end_idx) {
            return nullptr;
        }

        // 创建根节点，根节点的值使用前序遍历的第一个
        TreeNode* root = new TreeNode(preorder[pre_start_idx]);

        // 在中序遍历中找到根节点，划分为两个数组，分别是左右子树的，
        int root_idx = in_start_idx;
        for (; root_idx <= in_end_idx; root_idx++) {
            if (root->val == inorder[root_idx]) {
                break;
            }
        }

        // 左子树的长度
        int left_lens = root_idx - in_start_idx;

        // 创建左子树
        root->left = this->pre_inorder_build_tree(preorder, pre_start_idx + 1, pre_start_idx + left_lens, 
                                                  inorder, in_start_idx, root_idx - 1);
        // 创建右子树
        root->right = this->pre_inorder_build_tree(preorder, pre_start_idx + left_lens + 1, pre_end_idx, 
                                                  inorder, root_idx + 1, in_end_idx);

        return root;
    }
};
```
