// 定义链表节点类
class ListNode {
    int val; // 节点值
    ListNode next; // 指向下一个节点的指针

    ListNode(int val) {
        this.val = val;
        this.next = null;
    }
}
public class Solution {
    /**
     * 合并两个升序链表
     * @param l1 第一个升序链表的头节点
     * @param l2 第二个升序链表的头节点
     * @return 合并后的升序链表的头节点
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        // 创建一个虚拟头节点，方便操作
        ListNode dummy = new ListNode(-1);
        ListNode current = dummy;
        // 遍历两个链表，按顺序合并
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                current.next = l1; // 将 l1 当前节点接到结果链表
                l1 = l1.next; // 移动 l1 指针
            } else {
                current.next = l2; // 将 l2 当前节点接到结果链表
                l2 = l2.next; // 移动 l2 指针
            }
            current = current.next; // 移动结果链表的指针
        }
        // 如果 l1 或 l2 还有剩余节点，直接接到结果链表后面
        if (l1 != null) {
            current.next = l1;
        }
        if (l2 != null) {
            current.next = l2;
        }
        // 返回合并后的链表（跳过虚拟头节点）
        return dummy.next;
    }
    // 测试代码
    public static void main(String[] args) {
        // 创建测试链表 1->2->4
        ListNode l1 = new ListNode(1);
        l1.next = new ListNode(2);
        l1.next.next = new ListNode(4);

        // 创建测试链表 1->3->4
        ListNode l2 = new ListNode(1);
        l2.next = new ListNode(3);
        l2.next.next = new ListNode(4);

        // 合并链表
        Solution solution = new Solution();
        ListNode merged = solution.mergeTwoLists(l1, l2);

        // 打印结果链表
        while (merged != null) {
            System.out.print(merged.val + " ");
            merged = merged.next;
        }
    }
}