import type {
  LicenseConfig,
  NavBarConfig,
  ProfileConfig,
  SiteConfig,
} from './types/config'
import { LinkPreset } from './types/config'

export const siteConfig: SiteConfig = {
  title: 'h-terao\'s blog',
  subtitle: 'IT, Deep Learning, and more',
  lang: 'ja', // 'en', 'zh_CN', 'zh_TW', 'ja'
  themeColor: {
    hue: 250, // Default hue for the theme color, from 0 to 360. e.g. red: 0, teal: 200, cyan: 250, pink: 345
    fixed: true, // Hide the theme color picker for visitors
  },
  banner: {
    enable: false,
    src: 'assets/images/demo-banner.png', // Relative to the /src directory. Relative to the /public directory if it starts with '/'
  },
  favicon: [
    // Leave this array empty to use the default favicon
    {
      src: '/favicon/favicon.svg', // Path of the favicon, relative to the /public directory
      // theme: 'light',              // (Optional) Either 'light' or 'dark', set only if you have different favicons for light and dark mode
      // sizes: '32x32',              // (Optional) Size of the favicon, set only if you have favicons of different sizes
    },
  ],
}

export const navBarConfig: NavBarConfig = {
  links: [
    LinkPreset.Home,
    LinkPreset.Archive,
    LinkPreset.About,
    {
      name: 'GitHub',
      url: 'https://github.com/h-terao', // Internal links should not include the base path, as it is automatically added
      external: true, // Show an external link icon and will open in a new tab
    },
  ],
}

export const profileConfig: ProfileConfig = {
  avatar: 'assets/images/icon.png', // Relative to the /src directory. Relative to the /public directory if it starts with '/'
  name: 'h-terao',
  bio: '永遠の見習いITエンジニア',
  links: [
    // {
    //   name: 'Twitter',
    //   icon: 'fa6-brands:twitter',       // Visit https://icones.js.org/ for icon codes
    //                                     // You will need to install the corresponding icon set if it's not already included
    //                                     // `pnpm add @iconify-json/<icon-set-name>`
    //   url: 'https://twitter.com',
    // },
    // {
    //   name: 'GitHub',
    //   icon: 'fa6-brands:github',
    //   url: 'https://github.com/h-terao',
    // },
    // {
    //   name: 'Steam',
    //   icon: 'fa6-brands:steam',
    //   url: 'https://store.steampowered.com',
    // },
  ],
}

export const licenseConfig: LicenseConfig = {
  enable: true,
  name: 'CC BY-NC-SA 4.0',
  url: 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
}
