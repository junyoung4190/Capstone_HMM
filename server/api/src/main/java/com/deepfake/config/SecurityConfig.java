package com.deepfake.config;

import com.deepfake.jwt.JwtFilter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;

@EnableWebSecurity
@Configuration
public class SecurityConfig {

    private final JwtFilter jwtFilter;

    public SecurityConfig(JwtFilter jwtFilter) {
        this.jwtFilter = jwtFilter;
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {

        http
                .csrf(csrf -> csrf.disable())
                .httpBasic(basic -> basic.disable())
                .sessionManagement(session ->
                        session.sessionCreationPolicy(SessionCreationPolicy.STATELESS)
                )

                .authorizeHttpRequests(auth -> auth

                        // 회원가입 / 로그인 허용
                        .requestMatchers("/api/users/**").permitAll()

                        // 이미지 업로드 / 분석 / 보호 허용
                        .requestMatchers(
                                HttpMethod.POST,
                                "/api/images/upload",
                                "/api/images/analyze",
                                "/api/images/*/protect"
                        ).permitAll()

                        // 이미지 조회 및 파일 서빙 허용
                        .requestMatchers(
                                HttpMethod.GET,
                                "/api/images/**",
                                "/view/**"
                        ).permitAll()

                        // 나머지 인증 필요
                        .anyRequest().authenticated()
                )

                .addFilterBefore(
                        jwtFilter,
                        UsernamePasswordAuthenticationFilter.class
                );

        return http.build();
    }
}