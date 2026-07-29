if __name__=="__main__":
    import json
    from transformers import AutoConfig
    from simulator.core.model_analyzer import ModelAnalyzer
    from simulator.configs.models import llama
    model_name = "meta-llama/Llama-2-7b-hf"
    
    config = AutoConfig.from_pretrained(model_name)

    analyzer = ModelAnalyzer(
        model_id=model_name,
        config=llama,
        hardware="NVDA:H100:SXM",
    )
    results = analyzer.analyze(seqlen=10240, batchsize=1, w_bit=16, a_bit=16, kv_bit=16, tp_size=1)
    print(json.dumps(results, indent=2))
    gen_task_result = analyzer.analyze_generate_task(
        prompt_len=1024,
        gen_len=1024,
        batchsize=1,
        w_bit=16,
        a_bit=16,
        kv_bit=16,
        tp_size=1
    )
    print(json.dumps(gen_task_result, indent=2))