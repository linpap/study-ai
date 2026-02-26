'use client';

import { useRef, useCallback } from 'react';

interface CertificateProps {
  pathTitle: string;
  pathIcon: string;
  userEmail: string;
  completionDate: string;
  lessonsCompleted: number;
  exercisesCompleted: number;
  certificateId: string;
}

export default function Certificate({
  pathTitle,
  pathIcon,
  userEmail,
  completionDate,
  lessonsCompleted,
  exercisesCompleted,
  certificateId,
}: CertificateProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const generateCertificate = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = 1200;
    const h = 850;
    canvas.width = w;
    canvas.height = h;

    // Background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    // Border
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 4;
    ctx.strokeRect(20, 20, w - 40, h - 40);

    // Inner border
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 1;
    ctx.strokeRect(30, 30, w - 60, h - 60);

    // Corner decorations
    const cornerSize = 40;
    const corners = [
      [35, 35], [w - 35 - cornerSize, 35],
      [35, h - 35 - cornerSize], [w - 35 - cornerSize, h - 35 - cornerSize],
    ];
    corners.forEach(([x, y]) => {
      const gradient = ctx.createLinearGradient(x, y, x + cornerSize, y + cornerSize);
      gradient.addColorStop(0, '#3b82f6');
      gradient.addColorStop(1, '#8b5cf6');
      ctx.fillStyle = gradient;
      ctx.globalAlpha = 0.15;
      ctx.fillRect(x, y, cornerSize, cornerSize);
      ctx.globalAlpha = 1;
    });

    // StudyAI branding
    ctx.font = 'bold 28px system-ui, -apple-system, sans-serif';
    const brandGradient = ctx.createLinearGradient(w / 2 - 60, 80, w / 2 + 60, 80);
    brandGradient.addColorStop(0, '#3b82f6');
    brandGradient.addColorStop(1, '#8b5cf6');
    ctx.fillStyle = brandGradient;
    ctx.textAlign = 'center';
    ctx.fillText('StudyAI', w / 2, 90);

    // Certificate title
    ctx.font = '16px system-ui, -apple-system, sans-serif';
    ctx.fillStyle = '#6b7280';
    ctx.fillText('CERTIFICATE OF COMPLETION', w / 2, 130);

    // Divider
    const divGradient = ctx.createLinearGradient(w / 2 - 150, 0, w / 2 + 150, 0);
    divGradient.addColorStop(0, 'rgba(59, 130, 246, 0)');
    divGradient.addColorStop(0.5, '#3b82f6');
    divGradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
    ctx.strokeStyle = divGradient;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(w / 2 - 200, 150);
    ctx.lineTo(w / 2 + 200, 150);
    ctx.stroke();

    // "This certifies that"
    ctx.font = '18px system-ui, -apple-system, sans-serif';
    ctx.fillStyle = '#6b7280';
    ctx.fillText('This certifies that', w / 2, 210);

    // User email
    ctx.font = 'bold 32px system-ui, -apple-system, sans-serif';
    ctx.fillStyle = '#111827';
    ctx.fillText(userEmail, w / 2, 260);

    // "has completed"
    ctx.font = '18px system-ui, -apple-system, sans-serif';
    ctx.fillStyle = '#6b7280';
    ctx.fillText('has successfully completed the', w / 2, 320);

    // Path name with icon
    ctx.font = 'bold 40px system-ui, -apple-system, sans-serif';
    const pathGradient = ctx.createLinearGradient(w / 2 - 200, 380, w / 2 + 200, 380);
    pathGradient.addColorStop(0, '#3b82f6');
    pathGradient.addColorStop(1, '#8b5cf6');
    ctx.fillStyle = pathGradient;
    ctx.fillText(`${pathIcon} ${pathTitle}`, w / 2, 380);

    // "Learning Path"
    ctx.font = '20px system-ui, -apple-system, sans-serif';
    ctx.fillStyle = '#6b7280';
    ctx.fillText('Learning Path', w / 2, 420);

    // Stats
    ctx.font = '16px system-ui, -apple-system, sans-serif';
    ctx.fillStyle = '#374151';
    ctx.fillText(
      `${lessonsCompleted} lessons & ${exercisesCompleted} exercises completed`,
      w / 2,
      480
    );

    // Divider 2
    ctx.strokeStyle = divGradient;
    ctx.beginPath();
    ctx.moveTo(w / 2 - 200, 520);
    ctx.lineTo(w / 2 + 200, 520);
    ctx.stroke();

    // Date
    ctx.font = '16px system-ui, -apple-system, sans-serif';
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'left';
    ctx.fillText('Date:', 100, 600);
    ctx.fillStyle = '#111827';
    ctx.fillText(completionDate, 100, 625);

    // Issued by
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'right';
    ctx.fillText('Issued by:', w - 100, 600);
    ctx.fillStyle = '#111827';
    ctx.fillText('Greensolz', w - 100, 625);

    // Certificate ID
    ctx.font = '12px system-ui, -apple-system, sans-serif';
    ctx.fillStyle = '#9ca3af';
    ctx.textAlign = 'center';
    ctx.fillText(`Certificate ID: ${certificateId}`, w / 2, h - 50);

    // "Powered by StudyAI"
    ctx.fillText('learnai.greensolz.com', w / 2, h - 70);
  }, [pathTitle, pathIcon, userEmail, completionDate, lessonsCompleted, exercisesCompleted, certificateId]);

  const downloadCertificate = useCallback(() => {
    generateCertificate();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const link = document.createElement('a');
    link.download = `studyai-certificate-${pathTitle.toLowerCase().replace(/\s+/g, '-')}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  }, [generateCertificate, pathTitle]);

  return (
    <div>
      <canvas ref={canvasRef} className="hidden" />
      <button
        onClick={downloadCertificate}
        className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium rounded-lg hover:from-blue-600 hover:to-purple-700 transition text-sm"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        Download Certificate
      </button>
    </div>
  );
}
