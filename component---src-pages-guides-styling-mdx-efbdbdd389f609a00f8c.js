(window.webpackJsonp=window.webpackJsonp||[]).push([[47],{"013z":function(e,t,a){"use strict";var n=a("q1tI"),o=a.n(n),l=a("NmYn"),r=a.n(l),s=a("Wbzz"),i=a("Xrax"),c=a("k4MR"),b=a("TSYQ"),u=a.n(b),d=a("QH2O"),m=a.n(d),p=a("qKvR"),h=function(e){var t,a=e.title,n=e.theme,o=e.tabs,l=void 0===o?[]:o;return Object(p.b)("div",{className:u()(m.a.pageHeader,(t={},t[m.a.withTabs]=l.length,t[m.a.darkMode]="dark"===n,t))},Object(p.b)("div",{className:"bx--grid"},Object(p.b)("div",{className:"bx--row"},Object(p.b)("div",{className:"bx--col-lg-12"},Object(p.b)("h1",{id:"page-title",className:m.a.text},a)))))},j=a("BAC9"),g=function(e){var t=e.relativePagePath,a=e.repository,n=Object(s.useStaticQuery)("1364590287").site.siteMetadata.repository,o=a||n,l=o.baseUrl,r=o.subDirectory,i=l+"/edit/"+o.branch+r+"/src/pages"+t;return l?Object(p.b)("div",{className:"bx--row "+j.row},Object(p.b)("div",{className:"bx--col"},Object(p.b)("a",{className:j.link,href:i},"Edit this page on GitHub"))):null},O=a("FCXl"),x=a("dI71"),y=a("I8xM"),w=function(e){function t(){return e.apply(this,arguments)||this}return Object(x.a)(t,e),t.prototype.render=function(){var e=this.props,t=e.title,a=e.tabs,n=e.slug,o=n.split("/").filter(Boolean).slice(-1)[0],l=a.map((function(e){var t,a=r()(e,{lower:!0,strict:!0}),l=a===o,i=new RegExp(o+"/?(#.*)?$"),c=n.replace(i,a);return Object(p.b)("li",{key:e,className:u()((t={},t[y.selectedItem]=l,t),y.listItem)},Object(p.b)(s.Link,{className:y.link,to:""+c},e))}));return Object(p.b)("div",{className:y.tabsContainer},Object(p.b)("div",{className:"bx--grid"},Object(p.b)("div",{className:"bx--row"},Object(p.b)("div",{className:"bx--col-lg-12 bx--col-no-gutter"},Object(p.b)("nav",{"aria-label":t},Object(p.b)("ul",{className:y.list},l))))))},t}(o.a.Component),v=a("MjG9"),f=a("CzIb");t.a=function(e){var t=e.pageContext,a=e.children,n=e.location,o=e.Title,l=t.frontmatter,b=void 0===l?{}:l,u=t.relativePagePath,d=t.titleType,m=b.tabs,j=b.title,x=b.theme,y=b.description,N=b.keywords,T=Object(f.a)().interiorTheme,k=Object(s.useStaticQuery)("2456312558").site.pathPrefix,P=k?n.pathname.replace(k,""):n.pathname,C=m?P.split("/").filter(Boolean).slice(-1)[0]||r()(m[0],{lower:!0}):"",I=x||T;return Object(p.b)(c.a,{tabs:m,homepage:!1,theme:I,pageTitle:j,pageDescription:y,pageKeywords:N,titleType:d},Object(p.b)(h,{title:o?Object(p.b)(o,null):j,label:"label",tabs:m,theme:I}),m&&Object(p.b)(w,{title:j,slug:P,tabs:m,currentTab:C}),Object(p.b)(v.a,{padded:!0},a,Object(p.b)(g,{relativePagePath:u})),Object(p.b)(O.a,{pageContext:t,location:n,slug:P,tabs:m,currentTab:C}),Object(p.b)(i.a,null))}},BAC9:function(e,t,a){e.exports={bxTextTruncateEnd:"EditLink-module--bx--text-truncate--end--2pqje",bxTextTruncateFront:"EditLink-module--bx--text-truncate--front--3_lIE",link:"EditLink-module--link--1qzW3",row:"EditLink-module--row--1B9Gk"}},I8xM:function(e,t,a){e.exports={bxTextTruncateEnd:"PageTabs-module--bx--text-truncate--end--267NA",bxTextTruncateFront:"PageTabs-module--bx--text-truncate--front--3xEQF",tabsContainer:"PageTabs-module--tabs-container--8N4k0",list:"PageTabs-module--list--3eFQc",listItem:"PageTabs-module--list-item--nUmtD",link:"PageTabs-module--link--1mDJ1",selectedItem:"PageTabs-module--selected-item--YPVr3"}},QH2O:function(e,t,a){e.exports={bxTextTruncateEnd:"PageHeader-module--bx--text-truncate--end--mZWeX",bxTextTruncateFront:"PageHeader-module--bx--text-truncate--front--3zvrI",pageHeader:"PageHeader-module--page-header--3hIan",darkMode:"PageHeader-module--dark-mode--hBrwL",withTabs:"PageHeader-module--with-tabs--3nKxA",text:"PageHeader-module--text--o9LFq"}},s9d4:function(e,t,a){"use strict";a.r(t),a.d(t,"_frontmatter",(function(){return i})),a.d(t,"default",(function(){return d}));var n,o=a("wx14"),l=a("zLVn"),r=(a("q1tI"),a("7ljp")),s=a("013z"),i=(a("qKvR"),{}),c=(n="PageDescription",function(e){return console.warn("Component "+n+" was not imported, exported, or provided by MDXProvider as global scope"),Object(r.b)("div",e)}),b={_frontmatter:i},u=s.a;function d(e){var t=e.components,a=Object(l.a)(e,["components"]);return Object(r.b)(u,Object(o.a)({},b,a,{components:t,mdxType:"MDXLayout"}),Object(r.b)(c,{mdxType:"PageDescription"},Object(r.b)("p",null,"The carbon theme uses Sass to take advantage of the carbon-components styles and variables while authoring novel components. In addition, we use css modules to ensure we don’t collide with devs and make sure our components are portable and shadowable.")),Object(r.b)("h2",null,"Local Styles"),Object(r.b)("p",null,"For your application’s local styles, you can just import your style sheet ",Object(r.b)("a",{parentName:"p",href:"https://www.gatsbyjs.org/docs/global-css/#adding-global-styles-without-a-layout-component"},"directly into a ",Object(r.b)("inlineCode",{parentName:"a"},"gatsby-browser.js"))," file at the root of your project."),Object(r.b)("p",null,"You can also use sass modules like we do in the theme, this would make it easier for you to share your component with other theme consumers down the line."),Object(r.b)("p",null,"Every Sass file in your project automatically has access to the the following carbon resources:"),Object(r.b)("ul",null,Object(r.b)("li",{parentName:"ul"},"colors – ",Object(r.b)("inlineCode",{parentName:"li"},"background: carbon--gray-10;")),Object(r.b)("li",{parentName:"ul"},"spacing - ",Object(r.b)("inlineCode",{parentName:"li"},"margin: $spacing-05;")),Object(r.b)("li",{parentName:"ul"},"theme-tokens - ",Object(r.b)("inlineCode",{parentName:"li"},"color: $text-01;")),Object(r.b)("li",{parentName:"ul"},"motion - ",Object(r.b)("inlineCode",{parentName:"li"},"transition: all $duration--moderate-01 motion(entrance, productive);")),Object(r.b)("li",{parentName:"ul"},"layout - ",Object(r.b)("inlineCode",{parentName:"li"},"z-index: z('overlay');")),Object(r.b)("li",{parentName:"ul"},"typography - ",Object(r.b)("inlineCode",{parentName:"li"},"@include carbon--type-style('body-long-01');"))),Object(r.b)("h2",null,"Targeting theme components"),Object(r.b)("p",null,"Theme component classes have a hash of their styles tacked on to the end. This is both to prevent collisions, and also to prevent sneaky breaking changes to your styles if we update the class underneath you and you were relying on our styles."),Object(r.b)("p",null,"However, you can target the classes without the hash by using a ",Object(r.b)("a",{parentName:"p",href:"https://css-tricks.com/almanac/selectors/a/attribute/"},"partial selector"),":"),Object(r.b)("pre",null,Object(r.b)("code",{parentName:"pre",className:"language-scss"},"[class*='Banner-module--column'] {\n  color: $text-04;\n}\n")),Object(r.b)("p",null,"This will match the class that starts with ",Object(r.b)("inlineCode",{parentName:"p"},"Banner-module--column")," and would be immune to any changes to the hash. We may at some point remove the hash if this becomes an issue."))}d.isMDXComponent=!0}}]);
//# sourceMappingURL=component---src-pages-guides-styling-mdx-efbdbdd389f609a00f8c.js.map