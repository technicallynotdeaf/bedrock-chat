import { Popover, Transition } from '@headlessui/react';
import React, { Fragment, ReactNode, useMemo, useRef } from 'react';
import { createPortal } from 'react-dom';
import { PiDotsThreeOutlineFill } from 'react-icons/pi';

type Props = {
  className?: string;
  target?: 'bottom-left' | 'bottom-right';
  disabled?: boolean;
  children: ReactNode;
  portal?: boolean;
};

const PopoverMenu: React.FC<Props> = (props) => {
  const buttonRef = useRef<HTMLButtonElement>(null);

  const origin = useMemo(() => {
    if (props.target === 'bottom-left') {
      return 'left-0';
    } else if (props.target === 'bottom-right') {
      return 'right-0';
    }
    return 'left-3';
  }, [props.target]);

  return (
    <Popover className="relative">
      {({ open }) => {
        let portalStyle: React.CSSProperties = {};
        if (props.portal && open && buttonRef.current) {
          const rect = buttonRef.current.getBoundingClientRect();
          const spaceBelow = window.innerHeight - rect.bottom;
          const openUpward = spaceBelow < 200 && rect.top > 200;
          portalStyle = {
            position: 'fixed',
            zIndex: 50,
          };
          if (openUpward) {
            portalStyle.bottom = window.innerHeight - rect.top + 2;
          } else {
            portalStyle.top = rect.bottom + 2;
          }
          if (props.target === 'bottom-right') {
            portalStyle.right = window.innerWidth - rect.right;
          } else if (props.target === 'bottom-left') {
            portalStyle.left = rect.left;
          } else {
            portalStyle.left = rect.left + 12;
          }
        }

        const panel = (
          <Transition
            as={Fragment}
            enter="transition ease-out duration-200"
            enterFrom="opacity-0 translate-y-1"
            enterTo="opacity-100 translate-y-0"
            leave="transition ease-in duration-150"
            leaveFrom="opacity-100 translate-y-0"
            leaveTo="opacity-0 translate-y-1">
            <Popover.Panel
              className={props.portal ? '' : `absolute z-10 ${origin}`}
              style={props.portal ? portalStyle : undefined}>
              <div className="mt-0.5 overflow-hidden shadow-lg">
                <div className="flex flex-col whitespace-nowrap rounded border border-aws-font-color-light/50 bg-aws-paper-light text-sm dark:border-aws-font-color-dark/50 dark:bg-aws-paper-dark">
                  {props.children}
                </div>
              </div>
            </Popover.Panel>
          </Transition>
        );

        return (
          <>
            <Popover.Button
              ref={buttonRef}
              className={`${
                props.className ?? ''
              } group inline-flex items-center rounded-lg border border-aws-squid-ink-light/50 bg-aws-paper-light p-1 px-3 text-base hover:brightness-75 disabled:hover:brightness-100 dark:border-aws-font-color-gray/50 dark:bg-aws-paper-dark`}
              disabled={props.disabled}>
              <PiDotsThreeOutlineFill />
            </Popover.Button>
            {props.portal ? createPortal(panel, document.body) : panel}
          </>
        );
      }}
    </Popover>
  );
};

export default PopoverMenu;
