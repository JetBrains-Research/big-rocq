Theorem test : forall (A : Type) (P : A -> Prop) (x : A), P x -> P x.
Proof.
    auto.
Qed.

Theorem test2 : forall (A : Type) (P : A -> Prop) (x : A), P x -> P x.
Proof.
    intros A P x H.
    auto.
Qed.

Theorem test2nat1 : forall n : nat, n = 0 \/ n <> 0.
Proof.
  intros n.
  destruct n.
  { { left.
    reflexivity. } }
  right.
  discriminate.
Qed.

Theorem test2nat2 : forall n : nat, n = 0 \/ n <> 0.
Proof.
    intros n.
    destruct n.
    {
        auto.
    }
    {
        auto.
    }
Qed.

Theorem test_thr : forall n:nat, 0 + n = n.
Proof.
    intros n. Print plus.
    auto.
    (* reflexivity. *)
Qed.

Lemma test_thr1 : forall n:nat, 0 + n + 0 = n.
Proof.
    intros n. Print plus.
    auto.
    (* reflexivity. *)
Qed.
