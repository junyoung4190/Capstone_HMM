package com.deepfake.external;

import com.deepfake.dto.AnalyzeResult;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

@Component
public class FaceShieldClient {

    @Value("${ai.server.url}")
    private String aiServerUrl;

    private final RestTemplate restTemplate;

    public FaceShieldClient(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    // MultipartFile 대신 byte[] + filename을 받도록 변경
    // → @Async 컨텍스트에서도 안전하게 호출 가능
    public AnalyzeResult analyze(byte[] fileBytes, String originalName) {

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        ByteArrayResource fileResource =
                new ByteArrayResource(fileBytes) {
                    @Override
                    public String getFilename() {
                        return originalName;
                    }
                };

        MultiValueMap<String, Object> body =
                new LinkedMultiValueMap<>();
        body.add("file", fileResource);

        HttpEntity<MultiValueMap<String, Object>> request =
                new HttpEntity<>(body, headers);

        ResponseEntity<AnalyzeResult> response =
                restTemplate.postForEntity(
                        aiServerUrl + "/analyze",
                        request,
                        AnalyzeResult.class
                );

        if (response.getBody() == null) {
            throw new RuntimeException("AI 분석 실패");
        }

        return response.getBody();
    }

    public byte[] protect(byte[] fileBytes, String originalName) {

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        ByteArrayResource fileResource =
                new ByteArrayResource(fileBytes) {
                    @Override
                    public String getFilename() {
                        return originalName;
                    }
                };

        MultiValueMap<String, Object> body =
                new LinkedMultiValueMap<>();
        body.add("file", fileResource);

        HttpEntity<MultiValueMap<String, Object>> request =
                new HttpEntity<>(body, headers);

        ResponseEntity<byte[]> response =
                restTemplate.postForEntity(
                        aiServerUrl + "/protect",
                        request,
                        byte[].class
                );

        if (response.getBody() == null) {
            throw new RuntimeException("AI 보호 처리 실패");
        }

        return response.getBody();
    }
}