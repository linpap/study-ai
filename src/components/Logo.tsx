'use client';

import { useEffect, useRef } from 'react';
import Link from 'next/link';

interface LogoProps {
  size?: 'sm' | 'md' | 'lg';
  showText?: boolean;
  showTagline?: boolean;
}

export default function Logo({ size = 'md', showText = true, showTagline = false }: LogoProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const sizes = {
    sm: { container: 32, canvas: 32, text: 'text-lg', tagline: 'text-xs' },
    md: { container: 40, canvas: 40, text: 'text-xl', tagline: 'text-xs' },
    lg: { container: 56, canvas: 56, text: 'text-2xl', tagline: 'text-sm' },
  };

  const currentSize = sizes[size];

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const canvasSize = currentSize.canvas;

    canvas.width = canvasSize * dpr;
    canvas.height = canvasSize * dpr;
    canvas.style.width = `${canvasSize}px`;
    canvas.style.height = `${canvasSize}px`;
    ctx.scale(dpr, dpr);

    let animationId: number;
    let time = 0;

    const draw = () => {
      time += 0.02;
      ctx.clearRect(0, 0, canvasSize, canvasSize);

      // Animated glow intensity
      const glowIntensity = 0.5 + Math.sin(time) * 0.3;
      const pulseScale = 1 + Math.sin(time * 1.5) * 0.02;

      // Draw outer glow layers
      const centerX = canvasSize / 2;
      const centerY = canvasSize / 2;
      const baseRadius = (canvasSize / 2) - 4;

      // Multiple glow layers for depth
      for (let i = 3; i >= 0; i--) {
        const glowRadius = baseRadius + (i * 3);
        const alpha = glowIntensity * (0.15 - i * 0.03);

        const gradient = ctx.createRadialGradient(
          centerX, centerY, baseRadius * 0.5,
          centerX, centerY, glowRadius
        );
        gradient.addColorStop(0, `rgba(99, 102, 241, ${alpha})`);
        gradient.addColorStop(0.5, `rgba(139, 92, 246, ${alpha * 0.7})`);
        gradient.addColorStop(1, `rgba(168, 85, 247, 0)`);

        ctx.fillStyle = gradient;
        ctx.beginPath();
        roundedRect(ctx, centerX - glowRadius, centerY - glowRadius, glowRadius * 2, glowRadius * 2, 12 + i * 2);
        ctx.fill();
      }

      // Main rectangle with gradient
      const rectSize = baseRadius * 1.6 * pulseScale;
      const rectX = centerX - rectSize / 2;
      const rectY = centerY - rectSize / 2;

      const mainGradient = ctx.createLinearGradient(rectX, rectY, rectX + rectSize, rectY + rectSize);
      mainGradient.addColorStop(0, '#3B82F6');
      mainGradient.addColorStop(0.5, '#6366F1');
      mainGradient.addColorStop(1, '#8B5CF6');

      ctx.fillStyle = mainGradient;
      ctx.beginPath();
      roundedRect(ctx, rectX, rectY, rectSize, rectSize, 10);
      ctx.fill();

      // Inner shine effect
      const shineGradient = ctx.createLinearGradient(rectX, rectY, rectX + rectSize * 0.5, rectY + rectSize * 0.5);
      shineGradient.addColorStop(0, 'rgba(255, 255, 255, 0.25)');
      shineGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

      ctx.fillStyle = shineGradient;
      ctx.beginPath();
      roundedRect(ctx, rectX + 2, rectY + 2, rectSize * 0.6, rectSize * 0.4, 8);
      ctx.fill();

      // Draw "S" letter
      ctx.fillStyle = 'white';
      ctx.font = `bold ${rectSize * 0.55}px Inter, system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
      ctx.shadowBlur = 2;
      ctx.shadowOffsetY = 1;
      ctx.fillText('S', centerX, centerY + 1);
      ctx.shadowColor = 'transparent';

      // Particle effects
      drawParticles(ctx, centerX, centerY, rectSize, time);

      animationId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [currentSize.canvas]);

  return (
    <Link href="/" className="flex items-center gap-3 group">
      <div
        ref={containerRef}
        className="relative"
        style={{ width: currentSize.container, height: currentSize.container }}
      >
        <canvas
          ref={canvasRef}
          className="absolute inset-0"
        />
        {/* CSS glow layer for smoother effect */}
        <div
          className="absolute inset-0 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 opacity-0 group-hover:opacity-100 blur-md transition-opacity duration-300"
          style={{ transform: 'scale(1.2)' }}
        />
      </div>
      {showText && (
        <div>
          <span className={`${currentSize.text} font-bold bg-gradient-to-r from-blue-600 via-indigo-500 to-purple-600 bg-clip-text text-transparent`}>
            StudyAI
          </span>
          {showTagline && (
            <p className={`${currentSize.tagline} text-gray-500 dark:text-gray-400`}>
              Master AI & Machine Learning
            </p>
          )}
        </div>
      )}
    </Link>
  );
}

// Helper function to draw rounded rectangle
function roundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
) {
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

// Draw floating particles around the logo
function drawParticles(
  ctx: CanvasRenderingContext2D,
  centerX: number,
  centerY: number,
  size: number,
  time: number
) {
  const particleCount = 4;

  for (let i = 0; i < particleCount; i++) {
    const angle = (time * 0.5) + (i * Math.PI * 2 / particleCount);
    const distance = size * 0.7 + Math.sin(time * 2 + i) * 3;
    const x = centerX + Math.cos(angle) * distance;
    const y = centerY + Math.sin(angle) * distance;
    const particleSize = 1.5 + Math.sin(time * 3 + i) * 0.5;
    const alpha = 0.4 + Math.sin(time * 2 + i) * 0.3;

    const gradient = ctx.createRadialGradient(x, y, 0, x, y, particleSize * 2);
    gradient.addColorStop(0, `rgba(167, 139, 250, ${alpha})`);
    gradient.addColorStop(1, 'rgba(167, 139, 250, 0)');

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, particleSize * 2, 0, Math.PI * 2);
    ctx.fill();
  }
}
