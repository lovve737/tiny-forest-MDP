import mdptoolbox, mdptoolbox.example

P, R = mdptoolbox.example.forest()
print("전이확률\n",P)
print("\n보상\n",R)


"""정책 이터레이션"""
pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.95)
pi.run()

print("\n정책 이터레이션 밸류값\n",pi.V)
print("\n정책 이터레이션 정책\n",pi.policy)


"""밸류 이터레이션"""
vi = mdptoolbox.mdp.ValueIteration(P, R, 0.95)
vi.run()
vi.policy

print("\n밸류 이터레이션 정책\n",vi.policy)


"""Q러닝"""
ql = mdptoolbox.mdp.QLearning(P, R, 0.95)
ql.run()
ql.policy

print("\nQ러닝 밸류값\n",ql.V)
print("\nQ러닝 정책\n",ql.policy)


"""s3의 나무를 기다렸을때의 보상이 4 -> 0.4로 변경"""
P1, R1 = mdptoolbox.example.forest(3,0.4,2,0.1)
pi = mdptoolbox.mdp.PolicyIteration(P1, R1, 0.95)
pi.run()

print("\n보상이 변경된 정책 이터레이션 밸류값\n",pi.V)
print("\n보상이 변경된 정책 이터레이션 정책\n",pi.policy)


"""산불의 날 확률이 높은 경우, p=0.1 -> p=0.8"""
P2, R2 = mdptoolbox.example.forest(3,4,2,0.8)
pi = mdptoolbox.mdp.PolicyIteration(P2, R2, 0.95)
pi.run()

print("\n산불 확률이 변경된 정책 이터레이션 밸류값\n",pi.V)
print("\n산불 확률이 변경된 정책 이터레이션 정책\n",pi.policy)
