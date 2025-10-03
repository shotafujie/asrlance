def recognize_audio(audio_path, model_alias):
    if model_alias == "mlx":
        import mlx_whisper
        # Hugging Face repo名をモデル指定（ショートカットなら省略も可）
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            language="ja"
        )
        return result["text"]
    elif model_alias == "whisper":
        import whisper
        model = whisper.load_model("large-v3-turbo")
        result = model.transcribe(audio_path, language="ja")
        return result["text"]
    else:
        raise ValueError(f"Unknown model_alias: {model_alias}")

