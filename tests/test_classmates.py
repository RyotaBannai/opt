# from src.opt_classmates import run


# def test_model():
#     xs, ss, cs, result_df = run()
#     for s in ss:
#         assigned = [xs[s, c].value() for c in cs if xs[s, c].value() == 1]
#         assert len(assigned) == 1

#     for n in list(result_df.groupby("assigned_class")["student_id"].count()):
#         assert 39 <= n and n <= 40

# for n in list(result_df.groupby(["assigned_class", "gender"])["student_id"].count()):
# assert 39 <= n and n <= 40
