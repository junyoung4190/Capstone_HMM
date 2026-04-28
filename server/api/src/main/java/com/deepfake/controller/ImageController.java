package com.deepfake.controller;

import com.deepfake.dto.ImageResponse;
import com.deepfake.dto.ImageUploadResponse;
import com.deepfake.service.ImageService;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@RestController
@RequestMapping("/api/images")
public class ImageController {

    private final ImageService imageService;

    @Autowired
    public ImageController(ImageService imageService) {
        this.imageService = imageService;
    }

    @PostMapping("/upload")
    public ResponseEntity<ImageUploadResponse> upload(
            HttpServletRequest request,
            @RequestParam("file") MultipartFile file
    ) {
        Long userId = (Long) request.getAttribute("userId");
        return ResponseEntity.ok(imageService.uploadImage(file, userId));
    }

    @GetMapping("/my")
    public ResponseEntity<?> getMyImages(HttpServletRequest request) {
        Long userId = (Long) request.getAttribute("userId");

        if (userId == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body("로그인이 필요합니다.");
        }

        List<ImageResponse> images = imageService.getMyImages(userId);
        return ResponseEntity.ok(images);
    }
}