package com.deepfake.domain;

import com.deepfake.entity.ImageStatus;
import jakarta.persistence.*;

@Entity
@Table(name = "images")
public class Image {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String fileName;

    private String filePath;

    private String resultPath;

    private Double riskScore;

    @Column(length = 500)
    private String riskDescription; // ✅ 추가

    @Enumerated(EnumType.STRING)
    private ImageStatus status;

    @Column(length = 1000)
    private String errorMessage;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = true)
    private User user;

    public Image() {
    }

    public Image(
            String fileName,
            String filePath,
            User user
    ) {

        this.fileName = fileName;
        this.filePath = filePath;
        this.user = user;
        this.status = ImageStatus.PENDING;
    }

    public Long getId() {
        return id;
    }

    public String getFileName() {
        return fileName;
    }

    public String getFilePath() {
        return filePath;
    }

    public String getResultPath() {
        return resultPath;
    }

    public void setResultPath(String resultPath) {
        this.resultPath = resultPath;
    }

    public Double getRiskScore() {
        return riskScore;
    }

    public void setRiskScore(Double riskScore) {
        this.riskScore = riskScore;
    }

    public String getRiskDescription() { return riskDescription; } // ✅ 추가

    public void setRiskDescription(String riskDescription) { this.riskDescription = riskDescription; } // ✅ 추가

    public ImageStatus getStatus() {
        return status;
    }

    public void setStatus(ImageStatus status) {
        this.status = status;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }

    public User getUser() {
        return user;
    }
}