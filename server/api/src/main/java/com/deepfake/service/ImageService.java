package com.deepfake.service;

import com.deepfake.domain.Image;
import com.deepfake.domain.User;
import com.deepfake.dto.AnalyzeResult;
import com.deepfake.dto.ImageResponse;
import com.deepfake.dto.ImageUploadResponse;
import com.deepfake.entity.ImageStatus;
import com.deepfake.external.FaceShieldClient;
import com.deepfake.repository.ImageRepository;
import com.deepfake.repository.UserRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class ImageService {

    private final ImageRepository imageRepository;
    private final UserRepository userRepository;
    private final FaceShieldClient faceShieldClient;

    @Autowired
    public ImageService(
            ImageRepository imageRepository,
            UserRepository userRepository,
            FaceShieldClient faceShieldClient
    ) {
        this.imageRepository = imageRepository;
        this.userRepository = userRepository;
        this.faceShieldClient = faceShieldClient;
    }

    public ImageUploadResponse uploadImage(
            MultipartFile file,
            Long userId
    ) {

        try {

            User user = null;

            if (userId != null) {
                user = userRepository.findById(userId)
                        .orElseThrow(() ->
                                new RuntimeException("유저 없음"));
            }

            // 파일 바이트를 요청 컨텍스트 안에서 미리 추출
            // (MultipartFile은 @Async 컨텍스트에서 재사용 불가)
            byte[] fileBytes = file.getBytes();
            String originalName = file.getOriginalFilename();

            if (originalName == null || originalName.isBlank()) {
                throw new RuntimeException("파일 이름 없음");
            }

            // 파일 저장
            String savedName = saveFile(fileBytes, originalName);

            Image image = new Image(
                    originalName,
                    savedName,
                    user
            );

            image.setStatus(ImageStatus.PENDING);
            imageRepository.save(image);

            // AI 분석 — byte[]로 전달하여 비동기 컨텍스트에서도 안전하게 사용
            analyzeAsync(image.getId(), fileBytes, originalName);

            String url = "http://localhost:8080/view/" + savedName;

            return new ImageUploadResponse(
                    image.getId(),
                    url,
                    null
            );

        } catch (IOException e) {
            throw new RuntimeException("파일 저장 실패");
        }
    }

    @Async
    public void analyzeAsync(
            Long imageId,
            byte[] fileBytes,
            String originalName
    ) {

        Image image = imageRepository.findById(imageId)
                .orElseThrow(() ->
                        new RuntimeException("이미지 없음"));

        try {

            image.setStatus(ImageStatus.ANALYZING);
            imageRepository.save(image);

            AnalyzeResult result =
                    faceShieldClient.analyze(fileBytes, originalName);

            if (result.getRisk() != null) {
                image.setRiskScore(result.getRisk().getScore());
            }

            image.setStatus(ImageStatus.COMPLETED);
            imageRepository.save(image);

        } catch (Exception e) {

            image.setStatus(ImageStatus.FAILED);
            image.setErrorMessage(e.getMessage());
            imageRepository.save(image);

            e.printStackTrace();
        }
    }

    public List<ImageResponse> getMyImages(Long userId) {

        User user = userRepository.findById(userId)
                .orElseThrow(() ->
                        new RuntimeException("유저 없음"));

        return imageRepository.findByUser(user)
                .stream()
                .map(img -> new ImageResponse(
                        img.getId(),
                        img.getFileName(),
                        "http://localhost:8080/view/" + img.getFilePath(),
                        img.getStatus().name(),
                        img.getRiskScore(),
                        img.getResultPath(),
                        img.getErrorMessage()
                ))
                .collect(Collectors.toList());
    }

    public ImageResponse protectImage(Long imageId) {

        Image image = imageRepository.findById(imageId)
                .orElseThrow(() -> new RuntimeException("이미지 없음"));

        try {
            String uploadDir = System.getProperty("user.dir") + "/uploads/";
            byte[] fileBytes = Files.readAllBytes(
                    new File(uploadDir + image.getFilePath()).toPath()
            );

            byte[] protectedBytes = faceShieldClient.protect(fileBytes, image.getFileName());

            String savedName = saveFile(protectedBytes, "protected_" + image.getFileName());
            image.setResultPath(savedName);
            image.setStatus(ImageStatus.COMPLETED);
            imageRepository.save(image);

            return new ImageResponse(
                    image.getId(),
                    image.getFileName(),
                    "http://localhost:8080/view/" + image.getFilePath(),
                    image.getStatus().name(),
                    image.getRiskScore(),
                    "http://localhost:8080/view/" + savedName,
                    null
            );

        } catch (Exception e) {
            image.setStatus(ImageStatus.FAILED);
            image.setErrorMessage(e.getMessage());
            imageRepository.save(image);
            throw new RuntimeException("보호 처리 실패: " + e.getMessage());
        }
    }

    public ImageResponse getImageResult(Long imageId) {

        Image img = imageRepository.findById(imageId)
                .orElseThrow(() ->
                        new RuntimeException("이미지 없음"));

        return new ImageResponse(
                img.getId(),
                img.getFileName(),
                "http://localhost:8080/view/" + img.getFilePath(),
                img.getStatus().name(),
                img.getRiskScore(),
                img.getResultPath(),
                img.getErrorMessage()
        );
    }

    private String saveFile(byte[] fileBytes, String originalName)
            throws IOException {

        String uploadDir =
                System.getProperty("user.dir") + "/uploads/";

        File dir = new File(uploadDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        // 확장자 없는 파일명 방어 처리
        int dotIndex = originalName.lastIndexOf(".");
        String extension = (dotIndex >= 0)
                ? originalName.substring(dotIndex)
                : "";

        String savedName = UUID.randomUUID() + extension;
        File saveFile = new File(uploadDir + savedName);

        Files.write(saveFile.toPath(), fileBytes);

        System.out.println("저장 완료: " + saveFile.getAbsolutePath());
        System.out.println("저장 파일 크기: " + saveFile.length());

        return saveFile.getName();
    }
}