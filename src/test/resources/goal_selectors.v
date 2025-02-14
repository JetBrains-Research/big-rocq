Theorem no_selectors : forall (n m : nat), n + m = m + n.
Proof.
  intros n m.
  induction n.
  - simpl. auto.
  - simpl. rewrite IHn. auto.
Qed.

Theorem all_selector : forall (n : nat), 0 + n = n.
Proof.
  intros n.
  simpl.
  all: auto.
Qed.

Theorem select_nth : forall (n : nat), n * 0 = 0.
Proof.
  intros n.
  induction n.
  1: auto.
  auto.
Qed.

Theorem select_list : forall (n m p : nat), n + (m + p) = (n + m) + p.
Proof.
  intros n m p.
  induction n.
  1, 2: simpl; try rewrite IHn; reflexivity.
Qed.

Theorem select_range : forall (n : nat), n <> 0 -> pred (S n) = n.
Proof.
  intros n H.
  destruct n.
  1-2: try contradiction; try reflexivity.
Qed.

Inductive koma : Type :=
  | koma1 : koma
  | koma2 : koma
  | koma3 : koma.

Theorem complex_selector : forall (k : koma), k = koma1 \/ k = koma2 \/ k = koma3.
Proof.
  intros k.
  destruct k.
  1, 2-3: auto.
Qed.

Theorem parallel_selector : forall (n : nat), n <> 0 -> pred (S n) = n.
Proof.
  intros n H.
  destruct n.
  par: try contradiction; try reflexivity.
Qed.

Theorem single_goal : forall (n : nat), 0 + n = n.
Proof.
  intros n.
  simpl.
  (* Semantically equivalent to no selectors *)
  !: auto.
Qed.
