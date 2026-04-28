def protect_image(contents: bytes) -> bytes:
    # TODO: FaceShield — 적대적 perturbation 적용
    # return faceshield.process(contents)
    return contents  # mock: 원본 그대로 반환
