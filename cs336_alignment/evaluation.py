from vllm import LLM, SamplingParams
from cs336_alignment.utils.io_utils import LLAMA3_8B_BASE
from cs336_alignment.task import MMLU


def evaluate_mmlu():
    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
    temperature=0.0, top_p=1.0, max_tokens=1024, stop=["\n"])

    # Create an LLM.
    llm = LLM(model=LLAMA3_8B_BASE, device="cuda")
    
    mmlu = MMLU()
    prompts = [q.prompt for q in mmlu.questions]

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for question, output in zip(mmlu.questions, outputs):
        choice = MMLU.parse_mmlu_response(output.outputs[0].text)
        question.attempt = choice
        question.full_generation = output.outputs[0].text
    mmlu.save_attempts('out/mmlu.json')
    

if __name__ == "__main__":
    evaluate_mmlu()