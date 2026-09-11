'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navbar() {
  const pathname = usePathname();

  const navLinks = [
    { href: '/', label: 'Home' },
    { href: '/speech', label: 'Speech Clarity' },
    { href: '/pain', label: 'Pain Detection' },
    { href: '/about', label: 'About' },
  ];

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-40 shadow-xs">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
        <Link 
          href="/" 
          className="flex items-center gap-3 text-slate-900 font-bold text-xl hover:text-sky-700 transition-colors focus-visible:ring-2 focus-visible:ring-sky-600 rounded-lg p-1"
        >
          <span className="w-10 h-10 rounded-xl bg-sky-600 text-white flex items-center justify-center font-bold text-xl shadow-sm">
            AI
          </span>
          <div className="flex flex-col">
            <span className="leading-tight">Assistive AI</span>
            <span className="text-xs text-slate-500 font-normal">Speech & Pain Analysis</span>
          </div>
        </Link>

        <nav aria-label="Main Navigation">
          <ul className="flex items-center gap-1 sm:gap-2">
            {navLinks.map((link) => {
              const isActive = pathname === link.href;
              return (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    aria-current={isActive ? 'page' : undefined}
                    className={`px-3 py-2 sm:px-4 sm:py-2.5 rounded-lg text-sm sm:text-base font-semibold transition-all min-h-[44px] flex items-center ${
                      isActive
                        ? 'bg-sky-50 text-sky-700 border border-sky-200 shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                    }`}
                  >
                    {link.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
      </div>
    </header>
  );
}
