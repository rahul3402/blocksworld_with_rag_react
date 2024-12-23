test_plan = "(unstack d c)\n(put-down d)\n(pick-up c)\n(stack c a)\n"
with open("test_plan.txt", "w") as f:
    f.write(test_plan)

from response_evaluation import validate_plan

domain_file = "blocksworld/generated_domain.pddl"
domain_file = "./instances/blocksworld/generated_domain.pddl"
problem_file = "blocksworld/generated_basic/instance-2.pddl"
problem = "./instances/blocksworld/generated_domain.pddl"
plan_file = "test_plan.txt"

try:
    is_valid = validate_plan(domain_file, problem_file, plan_file)
    print(f"Plan Valid: {is_valid}")
except Exception as e:
    print(f"Validation Error: {e}")
